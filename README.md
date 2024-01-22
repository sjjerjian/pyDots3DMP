# pyDots3DMP

Python codes for dots3DMP experiments modelling and analysis.

### Basic Usage

1. Navigate to the project directory (pyDots3DMP) in your shell environment.
1. Install necessary packages in a virtual environment 
    - using venv
        - create virtual environment: ```$ python -m venv <your_env_name>```  
        - activate virtual environment: ```$ source <your_env_name>/bin/activate``` (MAC OS/Linux)
        - run ```$ pip install -r requirements.txt```
    - using conda
        - create environment (and install dependencies): ```$ conda create --name <your_env_name> --file environment.yml```
        - activate the environment: ```$ conda activate <your_env_name>```
1. Run a wrapper script e.g. ```dots3DMP_ddm_2d_example.py```, from the project directory (not the scripts sub-folder).

### ddm_2d design notes

Like standard optimization routines, BADS takes in an objective function to minimize, which should return a single output, the evaluated loss. It's other parameters are a vector array of initial parameters, and corresponding bounds on each parameter (plausible and hard bounds). Most of the handling around this in the code therefore is designed to allow the user to specify the parameter as key-value pairs, and to pass in the conditions list and observed behavior, accumulator design, and other flags/settings for likelihood evaluation.

The wrapper example script is ```dots3DMP_ddm_2d_example.py```, and the code relies on two sub-packages - ```ddm_moi``` (for handling the accumulator model, parameters, and defining the objective funciton) and ```behavior``` (for processing behavioral data and plotting results).

```ddm_moi``` is broken down into two submodules:
- ```ddm_2d```
    - The main workhorse here is ```generate_data```, which takes a set of parameters, a defined accumulator object, a conditions list, and other parameters, and returns model predictions of behavioural outcomes (choice, PDW, RT). The model predictions can either be probabilities, or simulated trials (either based on sampling from the probabilities, or simulating actual decision variables).

    - The model predictions are always returned as probability during optimization, but simulated trials can be used to generate model predictions with the final fit parameters, or to generate simulated data for testing parameter/model recovery.

    - To be used within the context of model evaluation, ```generate_data``` is called by an objective function, which returns the log likelihood of the actual observed outcomes in the data given the model parameters (and therefore the resulting predictions of the model).

    - The objective function is further wrapped for passing to the BADS object.
- ```Accumulator```
    - This handles all the low-level work of calculating the particles densities (CDFs, PDFs, log odds) that are needed for the model predictions, using the method of images with a two-dimensional accumulator model, and storing results within an Accumulator dataclass.


## Known issues, limitations and future improvements


1. Parameters specified as dictionary must be done so in a certain order right now, because transitioning from dictionary to array and back is not seamless. Dictionary is useful for readability and input flexibility in generate_data, but transitioning to and from array needs work to ensure robustness.
2. Fixed parameters are passed in as part of the same parameter list as the ones for fitting, and are then held constant on each iteration of the optimization (by being set to the fixed values before passing to the objective function). This does mean that BADS outputs contains useless values for these parameters, which then need to be overwritten with the original fixed values. It could make more sense to keep init_params and fixed as two separate dictionaries, and merge them in the ddm_2d module. This would also save having to set meaningless bounds on fixed parameters just for matching vector lengths.
2. There is some skeleton code logic for using ifferent cue combination strategies, and different confidence mappings, but this hasn't been tested.
3. ```generate_data``` could be doing too many things, and may be better broken down into one function for generating accumulator object and its attributes, and a second function for the model predictions (since these are independent of condition information, and just depend on the accumulator attributes).
4. Test different optimization routines (e.g. pyBADS vs scipy.minimize), and using multiple initial starting points.
5. Improve/expand documentation
6. Unit tests
