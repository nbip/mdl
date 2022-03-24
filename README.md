# Mixture of discretized logistic distributions

### Overview
Comparison of different implementations in `src/compare_mdl_implementations.py`. The different implementations are

- `mdl_openai.py`: Original OpenAI [PixelCNN++](https://github.com/openai/pixel-cnn) version.  
- `mdl_openai_wrapper.py`: Wrapper around then OpenAI version. Has the benefit of a `tfp.distributions`-like interface and much faster sampling.  
- `mdl_nbip.py`: A rewrite of the OpenAI PixelCNN from scratch. Has the benefit of easy customization, e.g. you can add a leading sample dimension to your parameters as in IWAEs.
- `mdl_vnca.py`: Wrapper around the [pytorch version of PixelCNN++](https://github.com/pclucas14/pixel-cnn-pp), copied from [VNCA](https://github.com/rasmusbergpalm/vnca).  

- `discretized_logistic.py`: Plain discretized logistic distribution.    
- `discretized_logistic_tfd.py`: Plain discretized logistic distribution, subclassing tfd.Distribution.  
- `mixture_discretized_logistic.py`: Plain mixture of discretized logistic distributions.   

### Resources:
[PixelCNN++](https://github.com/openai/pixel-cnn).  
[Variational Neural Cellular Automata](https://github.com/rasmusbergpalm/vnca).  
Subclassing tfd.Distribution: [tfp.distributions.Distribution](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution). See also 
[this notebook](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/jupyter_notebooks/Understanding_TensorFlow_Distributions_Shapes.ipynb) and the implementation of the [logistic distribution](https://github.com/tensorflow/probability/blob/v0.16.0/tensorflow_probability/python/distributions/logistic.py#L33-L236).

https://github.com/NVlabs/NVAE/blob/master/distributions.py  
https://github.com/openai/vdvae/blob/main/vae_helpers.py  
https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py  
https://github.com/JakobHavtorn/hvae-oodd/blob/main/oodd/layers/likelihoods.py  
https://arxiv.org/pdf/1701.05517.pdf  
https://github.com/rll/deepul/blob/master/demos/lecture2_autoregressive_models_demos.ipynb  
http://bjlkeng.github.io/posts/pixelcnn/  
https://bjlkeng.github.io/posts/autoregressive-autoencoders/  
https://bjlkeng.github.io/posts/importance-sampling-and-estimating-marginal-likelihood-in-variational-autoencoders/  
https://github.com/bjlkeng/sandbox/blob/master/notebooks/vae-importance_sampling/vae-cifar10-importance-sampling.ipynb  
https://github.com/didriknielsen/pixelcnn_flow/tree/master/pixelflow/transforms/functional  
