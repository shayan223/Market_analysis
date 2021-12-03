
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



candlestick_model = train_resnet_approximator(label_csv='./data/daily/candle_stick/labels.csv',
                                            data_dir='./data/daily/candle_stick/',
                                            batch_size=8,
                                            validation_split=.2,
                                            epochs=1)

# Create requisite directory structure
outdir ='./models/'
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)


#Save trained model weights
torch.save(candlestick_model.state_dict(), outdir+'candlestick.pt')



