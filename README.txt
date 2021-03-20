# Dependencies

	* Python 3.6

# Installation

	$ pip install -r requirement.txt
	$ cd model/modeling/DCNv2
	$ sh ./make.sh


# Dataset Preparetion

	Please write the directory path where you extracted the corresponding zip file for config/synthetic_config.yaml and config/real_config.yaml

# Training
	# Track 1

		$ python train.py --config_file config/synthetic_config.yaml

	# Track 2

		$ python train.py --config_file config/real_config.yaml

# Testing

	# Track 1

		$ python test.py --config_file config/synthetic_config.yaml

		test results will be saved in 'output/synthetic'

	# Track 2

		$ python test.py --config_file config/real_config.yaml

		test results will be saved in 'output/real'
