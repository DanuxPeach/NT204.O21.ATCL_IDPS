# %%
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
import random
from sklearn.metrics import classification_report
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from RITDS.util.masks import get_mask
import json
from RITDS.transformer import RTIDS_Transformer
from torch.optim import AdamW
import torch.nn as nn 
import logging

hyperParams = {
    'batchSize': 128,
    'randomState': 42,
    'validRatio': 0.33,
    'epochs': int(1e6), 
    'device': torch.device('cuda:0'),
    'numWorker': 8,
    'accumIter': 2,
    'verboseStep': 1,
    'minLr': 1e-4,
    'maxLr': 1e-3,
    'eps': 1e-8,
    'weightDecay': 1e-3,
    'modelDim': 32,
    'modelDepth': 5,
    'modelHead': 8
}


class CICIDS2017(Dataset):
    def __init__(self, data, label) -> None:
        super().__init__()
        self.data = data
        self.labels = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        target = self.labels[index]
        query = self.data[index]
        return query, target
def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up a logger."""
    handler = logging.FileHandler(log_file, encoding='utf-8')        
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

logger = setup_logger('training_logger', './training_log_2018.txt')

def initDataloader(dataPath, labelPath):
     data = np.load(dataPath)
     labels = np.load(labelPath)
     xTrain, xTest, yTrain, yTest = train_test_split(data, labels, test_size=hyperParams['validRatio'], random_state=hyperParams['randomState'])

     trainSet = CICIDS2017(xTrain, yTrain)
     validSet = CICIDS2017(xTest, yTest)

     trainLoader = torch.utils.data.DataLoader(trainSet,
                                               batch_size=hyperParams['batchSize'],
                                               num_workers=hyperParams['numWorker'],
                                               pin_memory=False,
                                               drop_last=True,
                                               shuffle=True)
     
     validLoader = torch.utils.data.DataLoader(validSet,
                                               batch_size=hyperParams['batchSize'],
                                               num_workers=hyperParams['numWorker'],
                                               pin_memory=False,
                                               drop_last=True,
                                               shuffle=True)
     
     return trainLoader, validLoader, data.shape[1]

# %%
class EarlyStopping:
    def __init__(self, datasetName, exportPath, patience=20, delta=0):
        self.report = None
        self.dtsName = datasetName
        self.exportPath = exportPath
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.bestLoss = None
        self.bestAcc = None
        self.earlyStop = False

    def __call__(self, val_loss, val_acc, model, report):
        
        if self.bestLoss is None:
            self.bestLoss = val_loss
            self.bestAcc = val_acc
            self.report = report

            torch.save({'dataset': self.dtsName,
                        'modelStateDict': model.state_dict(),
                        'bestLoss': self.bestLoss,
                        'bestAcc': self.bestAcc,
                        'report': self.report}, self.exportPath)


        elif val_loss > self.bestLoss + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            logger.info(f'Best loss is: {self.bestLoss}')
            logger.info(f'Best accuracy is: {self.bestAcc}')

            if self.counter >= self.patience:
                self.earlyStop = True
                
        else:
            self.bestLoss = val_loss
            self.bestAcc = val_acc
            self.report = report
            self.counter = 0

            torch.save({'dataset': self.dtsName,
                        'modelStateDict': model.state_dict(),
                        'bestLoss': self.bestLoss,
                        'bestAcc': self.bestAcc,
                        'report': self.report}, self.exportPath)


def seedEverything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

class ClearGPUMem:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def trainModel(epoch, model, lossFunc, optimizer, trainLoader, scheduler=None, schdBatchUpdate=False):
        model.train()
        runningLoss = None

        pbar = tqdm(enumerate(trainLoader), total=len(trainLoader))
        for step, (query, target) in pbar:
            query = query.to(hyperParams['device']).float()
            target = target.to(hyperParams['device']).long()

            scaler = GradScaler()
            with autocast():
                mask = get_mask(hyperParams['batchSize'], hyperParams['modelHead'], query.shape[1]).to(hyperParams['device'])
                predict = model(query, mask)
                
                loss = lossFunc(predict, target)
                scaler.scale(loss).backward()

                if runningLoss is None:
                    runningLoss = loss.item()
                else:
                    runningLoss = runningLoss * .99 + loss.item() * .01

                if ((step + 1) %  hyperParams['accumIter'] == 0) or ((step + 1) == len(trainLoader)):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad() 
                    
                    if scheduler is not None and schdBatchUpdate:
                        scheduler.step()

                if ((step + 1) % hyperParams['verboseStep'] == 0) or ((step + 1) == len(trainLoader)):
                    description = f'Epoch {epoch} loss: {runningLoss:.4f}'
                    pbar.set_description(description)
                  
        if scheduler is not None and not schdBatchUpdate:
           scheduler.step()

def evalModel(epoch, model, lossFn, validLoader, earlyStopping=None, labelmap=None, scheduler=None, schd_loss_update=False):      
        model.eval()

        lossSum = 0
        sampleNum = 0
        predictAll = []
        targetAll = []
        
        pbar = tqdm(enumerate(validLoader), total=len(validLoader))
        for step, (query, target) in pbar:
            query = query.to(hyperParams['device']).float()
            target = target.to(hyperParams['device']).long()
            
            predict = model(query)

            predictAll += [torch.argmax(predict, 1).detach().cpu().numpy()]
            targetAll += [target.detach().cpu().numpy()]
            
            loss = lossFn(predict, target)
            
            lossSum += loss.item() * target.shape[0]
            sampleNum += target.shape[0]

            if ((step + 1) % hyperParams['verboseStep'] == 0) or ((step + 1) == len(validLoader)):
                description = f'Epoch {epoch} loss: {lossSum/sampleNum:.4f}'
                pbar.set_description(description)
        
        predictAll = np.concatenate(predictAll)
        targetAll = np.concatenate(targetAll)

        report = classification_report(targetAll, predictAll, target_names=labelmap, digits=4)
        Loss = lossSum/sampleNum
        Acc = (predictAll==targetAll).mean()

        logger.info("---Classification Report---")
        logger.info(report)

        logger.info(f'Validating loss: {Loss}')
        logger.info(f'Validating accuracy: {Acc}')


        if earlyStopping != None:
            earlyStopping(Loss, Acc, model, report)
          
        if scheduler is not None:
            if schd_loss_update:
                scheduler.step(Loss)
            else:
                scheduler.step()

def main():
    
    seedEverything(hyperParams['randomState'])
    trainLoader, validLoader, numFeatures = initDataloader(dataPath='./preprocessedData/data.npy', labelPath='./preprocessedData/label.npy')
    labelMappingJson = json.load(open('./preprocessedData/labelMapping.json'))
    labelMapping = [label for label in labelMappingJson.values()]
    numClass = len(labelMapping)

    model = RTIDS_Transformer(numClass=numClass, dim=hyperParams['modelDim'], depth=hyperParams['modelDepth'], 
                            heads=hyperParams['modelHead'], maskSize=numFeatures).to(hyperParams['device'])

    with ClearGPUMem():
        optimizer = AdamW(model.parameters(), lr=hyperParams['maxLr'], eps=hyperParams['eps'], weight_decay=hyperParams['weightDecay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=hyperParams['minLr'])

        trainLossFunc = nn.CrossEntropyLoss().to(hyperParams['device'])
        validLossFunc = nn.CrossEntropyLoss().to(hyperParams['device'])
                            
        earlyStopping = EarlyStopping(datasetName='CICIDS2018', exportPath="./checkpoint/RITDS.checkpoint", patience=10)
        
        for epoch in range(hyperParams['epochs']):
            logger.info('=================================================')
            logger.info(f'\n[ TRAINING EPOCH {epoch} ]')
            TrainTime = trainModel(epoch, model, trainLossFunc, optimizer, 
                                trainLoader, scheduler, logger)
        
            with torch.no_grad():
                logger.info('\n[ EVALUATING VALIDATION ACCURACY ]')
                evalModel(epoch, model, validLossFunc, validLoader, 
                        earlyStopping, labelMapping, scheduler, logger)
                
                if earlyStopping.earlyStop:
                    logger.info("Early stopping triggered.")
                    break
    
    # Save the final model
    torch.save(model.state_dict(), './checkpoint/final_model.pth')
    logger.info("Final model saved.")

if __name__ == '__main__':
    # This is the recommended way to guard the entry point of the script.
    main()