import matplotlib.pyplot as plt
import numpy as np

## Generate an array of images from network output
def generate_images(network_output):
    num_graphs = network_output.shape[2]
    for i in range(num_graphs):
        plt.clf()
        plt.title(i)
        plt.imshow(network_output[:,:,i],cmap='gray')
        plt.axis('off')
        plt.draw()
        plt.pause(0.01)
        
        
##Create three plot comparisons on the output        
def generate_comparison_plot(volume, network_output, label, slice_num):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 20))
    ax1.imshow(np.squeeze(volume)[:,:,slice_num])
    ax1.set_title('Input Image')
    ax2.imshow(np.squeeze(network_output)[:,:,slice_num], cmap = 'gray')
    ax2.set_title('Prediction')
    ax3.imshow(np.squeeze(label)[:,:,slice_num], cmap = 'gray')
    ax3.set_title('Actual Lung Volume')
    fig.savefig('full_scan_prediction.png', dpi = 300)
    
    
    
## Plot the history graph from network output
def plot_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()