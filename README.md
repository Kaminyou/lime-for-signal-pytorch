# lime-for-signal-pytorch
LIME explainer for signal data (PyTorch version) </br>
This is an implementation of [Lime](https://arxiv.org/abs/1602.04938) for signal data.

## Requirement
Please make sure you have installed all the following package before using this tool.
1. Numpy
2. PyTorch
3. Seaborn
4. Matplotlib
5. Scikit-learn

## Tutorial
### Initialize
The **LimeSignal** class should be initialized first. You must provide: 
1. **x**: the signal data in a numpy (Channel * Dimension) array. It is recommended that the data are scaled in (-1, 1) interval.
2. **ground_true**: int value
3. **model**: your multi-class classifier pytorch model and the output should be in a torch vector that match each class 
4. **device**: "cpu" or "cuda"
```py
from limesignal import LimeSingal 
explainer = LimeSignal(x, ground_true, model, device)
```
The dafault partition is dependent on the cumulative value of the sequential signal in your data. After the cumulative value over *cumulative_criteria*, the cumulative value will initialize to 0 and separate the previous part. </br>
You can set the *cumulative_criteria* by:
```py
from limesignal import LimeSingal 
explainer = LimeSignal(x, ground_true, model, device, cumulative_criteria = 2)
```
You can provide your self-defined partition but please make sure the dimension must be the same as your input data
```py
from limesignal import LimeSingal 
explainer = LimeSignal(x, ground_true, model, device, x_partition = YOUR_PARTITION)
```
Other parameters include: drop_ratio, iteration, and lasso_alpha. You can set them by
```py
from limesignal import LimeSingal 
explainer = LimeSignal(x, ground_true, model, device, drop_ratio = 0.2, iteration = 100, lasso_alpha=0.000001)
```
### Get the weight
```py
weight_for_each_partition = explainer.output_weight()
```

### Plot
```py
explainer.show_line_plot(channel_name_list=[name_for_channal_one, name_for_channal_two, ...])
```
or 
```py
explainer.show_scatter_plot(channel_name_list=[name_for_channal_one, name_for_channal_two, ...])
```

