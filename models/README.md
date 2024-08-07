* code of model training will be stored here.
* need to get a token from [huggingface](https://huggingface.co/settings/tokens)
* **current blocker: setup GPU**



### codes
* set_env.sh - ref [huggingface environment variables](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)
  - This code exports variables for huggingface models to store cache files.
  - need to specify to scratch space to avoid home directory disk quota exceed.
  - also need to specify the path to token file to log in later
  - after `source set_env.sh`, login to huggingface-cli with command `huggingface-cli login`
 
* utils.py
   - functions reads the input file including texts and annotations
   - functions to tokenize the input texts and annotations with AutoTokenizer from pretrained models; pretrained models load remotely
   - functions for model fine-tuning and prompting
   - functions for model performance evaluation
   - functions for output
 
* models.py
   - use the functions in utils.py and train models.
