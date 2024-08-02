# use python 
FROM python:3.11

# set the working directory
WORKDIR /

# copy requirements.txt into the container
COPY requirements.txt .

# install the needed packages
RUN pip install --no-cache-dir -r requirements.txt 

# copy current dir contents into the container
COPY . .

# Run the application
# for training the model and plot the model performance 
CMD ["python", "inference_model.py"]

# for interpreting the models
# CMD ["python", "inference_interpret.py"]