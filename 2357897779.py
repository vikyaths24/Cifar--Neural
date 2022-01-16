from matplotlib import pyplot as plt
import numpy as np
import pickle

def get_data(inputs_file_path):
    """
    Takes in an inputs file path and labels file path, loads the data into a dict, 
    normalizes the inputs, and returns (NumPy array of inputs, NumPy 
    array of labels). 
    :param inputs_file_path: file path for ONE input batch, something like 
    'cifar-10-batches-py/data_batch_1'
    :return: NumPy array of inputs as float32 and labels as int8
    """
    #TODO: Load inputs and labels
    with open(inputs_file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    
    #TODO: Normalize inputs
    input=dict[b'data']/255
    input=input.astype('float32')
    encodingtrainlabel=np.zeros((len(dict[b'labels']),10),dtype='int8')
    i=0
    
    for k in dict[b'labels']:
      
      encodingtrainlabel[i][int(k)]=1
      i+=1
    
    return input,encodingtrainlabel,dict[b'labels']
class Model:
    """
    This model class will contain the architecture for
    your single layer Neural Network for classifying CIFAR10 with 
    batched learning. Please implement the TODOs for the entire 
    model but do not change the method and constructor arguments. 
    Make sure that your Model class works with multiple batch 
    sizes. Additionally, please exclusively use NumPy and 
    Python built-in functions for your implementation.
    """

    def __init__(self):
        # TODO: Initialize all hyperparametrs
        self.input_size = 3072 # Size of image vectors
        self.num_classes = 10 # Number of classes/possible labels
        self.batch_size =2
        self.learning_rate = 0.7326892

        # TODO: Initialize weights and biases
        self.W = np.zeros((10,3072))
        self.b = np.zeros((10,1))

    def forward(self, inputs):
        """
        Does the forward pass on an batch of input images.
        :param inputs: normalized (0.0 to 1.0) batch of images,
                       (batch_size x 3072) (2D), where batch can be any number.
        :return: probabilities, probabilities for each class per image # (batch_size x 10)
        """
        # TODO: Write the forward pass logic for your model
        self.fp=np.dot(self.W,inputs)+self.b
        # TODO: Calculate, then return, the probability for each class per image using the Softmax equation
        X=self.fp.T
    
        max = np.max(X,axis=1,keepdims=True) 
        e_x = np.exp(X - max) 
        sum = np.sum(e_x,axis=1,keepdims=True) 
        soft = e_x / sum 
    
        
        return soft.T
        pass
    
    def loss(self, probabilities, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Loss should be generally decreasing with every training loop (step). 
        :param probabilities: matrix that contains the probabilities 
        of each class for each image
        :param labels: the true batch labels
        :return: average loss per batch element (float)
        """
        # TODO: calculate average cross entropy loss for a batch
        sampleno=labels.shape[1]
        
        self.loss2=-(1./sampleno)*np.sum(labels*np.log(probabilities))
        return self.loss2
        pass
    
    def compute_gradients(self, inputs, probabilities, labels):
        """
        Returns the gradients for model's weights and biases 
        after one forward pass and loss calculation. You should take the
        average of the gradients across all images in the batch.
        :param inputs: batch inputs (a batch of images)
        :param probabilities: matrix that contains the probabilities of each 
        class for each image
        :param labels: true labels
        :return: gradient for weights,and gradient for biases
        """
        # TODO: calculate the gradients for the weights and the gradients for the bias with respect to average loss
        dcost=probabilities-labels
        size=inputs.shape[1]
        dw=1./size*np.dot(dcost,inputs)
        db=1./size*np.sum(dcost,axis=1,keepdims=True)
        return dw,db
        pass
    
    def accuracy(self, probabilities, labels):
        """
        Calculates the model's accuracy by comparing the number 
        of correct predictions with the correct answers.
        :param probabilities: result of running model.forward() on test inputs
        :param labels: test set labels
        :return: Float (0,1) that contains batch accuracy
        """
        # TODO: calculate the batch accuracy
        l=probabilities.T
        right=0
        predicted_labels = np.argmax(probabilities, axis=1)
        
        right=0
        for i in range (l.shape[0]):
          max=-np.inf
          for j in range (l.shape[1]):
            if l[i][j]>max:
              max=l[i][j]
              predict=j
          if int(labels[i])==int(predict):
            
            right+=1
         
        
        return right/(l.shape[0])
        
        pass

    def gradient_descent(self, gradW, gradB):
        '''
        Given the gradients for weights and biases, does gradient 
        descent on the Model's parameters.
        :param gradW: gradient for weights
        :param gradB: gradient for biases
        :return: None
        '''
        # TODO: change the weights and biases of the model to descent the gradient
        self.W=self.W-self.learning_rate*gradW
        self.b=self.b-self.learning_rate*gradB
        pass
    
def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels.
    :param model: the initialized model to use for the forward 
    pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training)
    :param train_inputs: train labels (all labels to use for training)
    :return: None
    '''
    loss=[]
    k=0
    n=model.batch_size
    # TODO: Iterate over the training inputs and labels, in model.batch_size increments
    for f in range(int(train_inputs.shape[0]/model.batch_size)):
      train_inputs2=train_inputs[k:n,:]
      train_labels2=train_labels[k:n,:]
        
      prob=model.forward(train_inputs2.T)
      
    # TODO: For every batch, compute then descend the gradients for the model's weights
      dw,db=model.compute_gradients(train_inputs2,prob,train_labels2.T)
      model.gradient_descent(dw,db)
      k=int(k+(model.batch_size))
      n=int(n+(model.batch_size))
      loss.append(model.loss(prob,train_labels2.T))
    if train_inputs.shape[0]%model.batch_size!=0:    
      train_inputs2=train_inputs[k:train_inputs.shape[0],:]
      train_labels2=train_labels[k:train_inputs.shape[0],:]
        
      prob=model.forward(train_inputs2.T)
    
    # TODO: For every batch, compute then descend the gradients for the model's weights
      dw,db=model.compute_gradients(train_inputs2,prob,train_labels2.T)
      model.gradient_descent(dw,db)
      loss.append(model.loss(prob,train_labels2.T))    
    # Optional TODO: Call visualize_loss and observe the loss per batch as the model trains.
    
    visualize_loss(loss)
    pass

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. 
    :param test_inputs: CIFAR10 test data (all images to be tested)
    :param test_labels: CIFAR10 test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """
    # TODO: Iterate over the testing inputs and labels
    prob=model.forward(test_inputs.T)
   
    # TODO: Return accuracy across testing set
    acc=model.accuracy(prob,test_labels)
    return acc
    pass

def visualize_loss(losses):
    """
    Uses Matplotlib to visualize loss per batch. You can call this in train() to observe.
    param losses: an array of loss value from each batch of train

    NOTE: DO NOT EDIT
    
    :return: doesn't return anything, a plot should pop-up
    """

    plt.ion()
    plt.show()

    x = np.arange(1, len(losses)+1)
    plt.xlabel('i\'th Batch')
    plt.ylabel('Loss Value')
    plt.title('Loss per Batch')
    plt.plot(x, losses, color='r')
    plt.draw()
    plt.pause(0.001)

def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data()
    :param probabilities: the output of model.forward()
    :param image_labels: the labels from get_data()

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    plt.ioff()

    images = np.reshape(image_inputs, (-1, 3, 32, 32))
    images = np.moveaxis(images, 1, -1)
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = images.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()

def main():
    '''
    Read in CIFAR10 data, initialize your model, and train and test your model 
    for one epoch. The number of training steps should be your the number of 
    batches you run through in a single epoch. 
    :return: None
    '''
   
    # TODO: load CIFAR10 train and test examples into train_inputs, train_labels, test_inputs, test_labels
    
    train_inputs,train_labels,check=get_data('data_batch_1')
    test_inputs,test_labels,checktest=get_data('test_batch')
    for i in range(2,6):
      train_inputsh,train_labelsh,checkh=get_data('data_batch_'+str(i))
      train_inputs=np.append(train_inputs,train_inputsh,axis=0)
      train_labels=np.append(train_labels,train_labelsh,axis=0)
      check=np.append(check,checkh)
    
    # TODO: Create Model
    model=Model()
    # TODO: Train model by calling train() ONCE on all data
    train(model,train_inputs,train_labels)
    # TODO: Test the accuracy by calling test() after running train()
    
    acc=test(model,test_inputs,checktest)
    print(f"{acc:.4f}")
    prob=model.forward(test_inputs.T)
    # TODO: Visualize the data by using visualize_results() on a set of 10 examples
    k=prob.T
    visualize_results(test_inputs[0:10],k[0:10],checktest[0:10])
    pass
    
if __name__ == '__main__':
    main()