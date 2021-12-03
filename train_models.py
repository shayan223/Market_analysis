
from value_approximator_1 import train_resnet_approximator
import os
import torch


'''For reference, use this to save/load

#NOTE SAVING THIS WAY COULD CAUSE PROBLEMS, THE MODEL ABOVE IS BOUND TO DIRECTORY STRUCTURE

torch.save(model, PATH)

# Model class must be defined somewhere
model = torch.load(PATH)
model.eval()


###########
OR this
###########

#NOTE NOT USING THE BELOW STATE DICT COULD CAUSE PROBLEMS, THE MODEL ABOVE IS BOUND TO DIRECTORY STRUCTURE

torch.save(model.state_dict(), PATH)

model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
'''


'''###############   Parameters   ##############################'''

VAL_SPLIT = .2
BATCH_SIZE = 4
EPOCHS = 2
LR = .01

'''###############################################################'''
def run_training(experiment_name):
    # Create requisite directory structure
    outdir ='./models/'+str(experiment_name)+'/'
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    fit_model = train_resnet_approximator(model_name=str(experiment_name),
                                                label_csv='./data/daily/'+str(experiment_name)+'/labels.csv',
                                                data_dir='./data/daily/'+str(experiment_name)+'/',
                                                out_dir=outdir,
                                                batch_size=BATCH_SIZE,
                                                validation_split=VAL_SPLIT,
                                                epochs=EPOCHS,
                                                lr=LR)



    #Save trained model weights
    torch.save(fit_model.state_dict(), outdir+str(experiment_name)+'.pt')
    return fit_model


run_training('candle_stick')
run_training('movingAvg')
run_training('PandF')
run_training('price_line')
run_training('renko')