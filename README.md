# Overview

Original OpenAI PixelCNN version in `dmol_openai.py`.  
Wrapper around then OpenAI version in `dmol_openai_wrapper.py`.  
A rewrite of the OpenAI PixelCNN from scratch in `dmol_nbip.py`  

Wrapper around the pytorch version of PixelCNN in `dmol_vnca.py` (from [VNCA](https://github.com/rasmusbergpalm/vnca))  

Plain discretized logistic distribution in `discretized_logistic.py`.    
Plain mixture of discretized logistic distributions in `mixture_discretized_logistic.py`.  
Plain discretized logistic distribution, subclassing tfd.Distribution in `dmol_tfd.py`.   

# Resources:
Subclassing tfd.Distribution: [tfp.distributions.Distribution](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution). See also 
[this notebook](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/jupyter_notebooks/Understanding_TensorFlow_Distributions_Shapes.ipynb) and the implementation of the [logistic distribution](https://github.com/tensorflow/probability/blob/v0.16.0/tensorflow_probability/python/distributions/logistic.py#L33-L236).
