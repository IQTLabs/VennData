import matplotlib.pyplot as plt
import numpy as np

def timeSeries(r_null,r_detect):
    
    # Function: timeSeries
    # Inputs:   r_null   (numpy array) size=(number of batches)
    #           r_detect (numpy array) size=(number of batches)
    # Process: plots time series of detection signal
    # Output:   none
    
    line_null,   = plt.plot(r_null, label='No Label Shift')
    line_detect, = plt.plot(r_detect, label='Label Shift')
    
    plt.legend(handles=[line_null, line_detect])
    plt.show()
    
def plotHistogram(r_null,r_detect,max_val):

    # Function: plotHistogram
    # Inputs:   r_null   (numpy array) size=(number of batches)
    #           r_detect (numpy array) size=(number of batches)
    #           max_val  (int)
    # Process: plots histograms of detection signal
    # Output:   none
    
    bins = np.linspace(0,max_val,100)

    hist_no_shift, _ = np.histogram(r_null, bins=bins, density=True)
    hist_shift, _    = np.histogram(r_detect, bins=bins, density=True)

    line_null,   = plt.plot(bins[0:99],hist_no_shift, label='No Label Shift')
    line_detect, = plt.plot(bins[0:99],hist_shift, label='Label Shift')

    plt.legend(handles=[line_null, line_detect])
    plt.show()