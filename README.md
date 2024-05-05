# What's this repo?
This repo uses [Yisol's Virtual Try On](https://huggingface.co/spaces/yisol/IDM-VTON) and enhance's it by cleaning the code, solving some errors and building an api around it
# How to set it up?
Make sure to have a valid python 3.10 installation in your system
```bash
# Clone original repo
git clone https://huggingface.co/spaces/yisol/IDM-VTON
cd IDM-VTON
# Install git lfs files
git lfs pull

# Clone our repo in another folder
cd ..
git clone https://github.com/HACKUPC-Try-on/backend-tryon-inference

# Install our new files over the other package
cp backend-tryon-inference/* IDM-VTON/
# Make sure to have poetry installed and ready on your machine
poetry install --no-root
# Now the app script has been substituted by our own version and you just need to run it
poetry run python main.py
```
That's it! Our API is running on localhost, on port 8000!

# Are you a developer?
If you want to contribute to our project, feel free to do!
Make sure to check out our `CONTRIBUTING.md` file