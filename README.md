# Ticket bot (搶票機器人)

## Installation
```ruby
pip install -r requirements.txt
```
If you are using pytorch for captcha model and use cpu only:
```ruby
pip install -r requirements_torch.txt
```
If you are using gpu, please check https://pytorch.org/get-started/previous-versions/ and find the corresponding torch version for installation.

## Getting started
### Step1
Please find your MS Edge browser path and paste into the $browser_path within bot.py.
### Step2 
Please refer to run.ipynb

## Captcha model
### Model structure
ResNet18 + FPN + Transformer
### Traning strategy
Model is first trained on 20k captcha data generate by [captcha](https://pypi.org/project/captcha/) library, and then fine-tuned on 550 tixcraft captcha data. Model is trained using cross entropy loss with a batch size of 50 for 20 epochs and optimized by AdamW with learning rate 1e-3. For fine-tuning, model is trained for 500 epochs with batch size of 20 and learning rate is set to 2e-4.
### Accuracy on tixcraft validation data 
![](tixcraft_acc.png)
