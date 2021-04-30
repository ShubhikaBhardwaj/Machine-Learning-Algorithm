**<h1>Capstone Project- Music Generation</h1>**


![alt text](https://data.whicdn.com/images/139175288/original.jpg)

**<h2>About the Project</h2>**

In this project, we generated unique melodious music from piano notes & chords. This is achieved by implementing a [Deep Learning Algorithm](https://towardsdatascience.com/what-is-deep-learning-and-how-does-it-work-2ce44bb692ac) called **Recurrent Neural Networks with LSTM** and
**Dense layers** for sequential music generation.
  
Here, we have employed a vast training data set consisting of **60,498** music
elements and **5,474,663** trainable parameters. 

The training data is required to be in MIDI format for convienent parsing with the help of Music21 Library.



**<h2>How to use</h2>**

- Step 1. Install Python 3.6.8 
- Step 2. Set up your IDE for Python (preferrably Jupyter Notebook/Google Colab)
- Step 3. Install the required Libraries using pip
- Step 4. Extract the melodious songs from the [midi_songs.zip](https://github.com/ShubhikaBhardwaj/Machine-Learning-Algorithm/blob/master/Capstone%20Project-%20Music%20Generation/midi_songs.zip) This is our training data
- Step 5. Train the Deep Learning model using RNN Algorithm with LSTM layers. 
          Refer to [Music_Generation.ipynb](https://github.com/ShubhikaBhardwaj/Machine-Learning-Algorithm/blob/master/Capstone%20Project-%20Music%20Generation/Music%20Generation.ipynb) for the RNN Algorithm's implementation
          If you are using Google Colab,refer to [Music_Generation_Using_Colab.ipynb](https://github.com/ShubhikaBhardwaj/Machine-Learning-Algorithm/blob/master/Capstone%20Project-%20Music%20Generation/Music_Generation_Using_Colab.ipynb)
- Step 6. The weights from the trained model are stored as [new_weights.mdf5](https://github.com/ShubhikaBhardwaj/Machine-Learning-Algorithm/blob/master/Capstone%20Project-%20Music%20Generation/new_weights.hdf5)
- Step 7. Now, you can generate your own unique melodious music using the trained model
          The new song generated using Deep Learning algorithms is [test_output.mid](https://github.com/ShubhikaBhardwaj/Machine-Learning-Algorithm/blob/master/Capstone%20Project-%20Music%20Generation/test_output.mid)
- Step 8. Have fun and enjoy your new song! 
         

