# Carnatic-Violin-FTA-ResNet
 This is the repository for Carnatic Violin pitch tracking using the FTA-ResNet architecture

Follow these steps to use this model for your research purposes -
1. Clone this repository using `git clone` command
2. Change the current directory to this one
3. Create a virtual environment and activate it ([Guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/))
4. Run `pip install -r requirements.txt` to install all the packages

## Inference
 To use the model for inference purposes, run the `inference.py` from your terminal as follows

 ```
python inference.py "path_to_wav_or_mp3_file"
 ```

The code shall create plots and resynthesized audio files in the `results` folder
