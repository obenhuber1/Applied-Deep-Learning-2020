# Imports for File Path Management and Keras Components
import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Path Management and Nb of Samples for Productive Setup
os.chdir("C:/TU/Deep Learning/Simba") # Main Directory, adapt accordingly to your File System Structure
working_dir = os.path.join(os.getcwd(),'images_model')
validate_data_dir = os.path.join(working_dir,"validation")
test_data_dir = os.path.join(working_dir,"test")

# Static Variables
img_width, img_height = 300, 300 # Standard Image size
color_channels = 3 # Number of used color channels

# Definition of Test Generator
test_data_generator = ImageDataGenerator(
    rescale = 1./255
    )

test_gen = test_data_generator.flow_from_directory(
    test_data_dir,
    target_size=(img_height,img_width),
    class_mode = 'categorical',
    batch_size=1,
    shuffle = False
    )


### Code in case you want to load a specific model
# Out of Sample Evaluation on completely unseen data
model_name = 'simba_vgg3_final_more_simba' # Folder and Name of the model you want to load (both are the same string)
model = load_model('model_outputs/ ' + model_name+ '/' + model_name + '.npz')
model_name_dir ='simba_vgg3_final_predictions' # All Outputs will be put into a subfolder with this string

# Create Confusion Matrix for Test Sample and save Plots
pre_class_oos = model.predict_generator(test_gen) # Load an existing model into variable 'model'
labels_oos = test_gen.classes
pre_class_one_hot_oos = pre_class_oos.argmax(1)
cm_rel_oos = confusion_matrix(labels_oos, pre_class_one_hot_oos, normalize='true')

class_names_oos = test_gen.class_indices
cm2_rel = ConfusionMatrixDisplay(cm_rel_oos, display_labels=class_names_oos)
cm2_rel.plot()
cm2_rel.figure_.savefig('model_outputs/' + model_name + '/cm_rel_test_samples.png')