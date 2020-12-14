# Imports for File Path Management, Plotting, Confusion Matrix and Keras Components
import os

from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.callbacks import  EarlyStopping, ModelCheckpoint, LearningRateScheduler, CSVLogger
from keras.layers import Conv2D, MaxPooling2D # Special Layers
from keras.layers import Activation, BatchNormalization, Flatten, Dense # Standard Layer
from keras.optimizers import RMSprop

# Static Variables
img_width, img_height = 300, 300 # Standard Image size
color_channels = 3 # Number of used color channels

model_name ='simba_vgg3_final_more_simba' # All Outputs will be put into a subfolder with this string


# Path Management
os.chdir("C:/github_simba_ex2/") # Main Directory, adapt accordingly to your File System Structure
working_dir = os.path.join(os.getcwd(),'images_model')
validate_data_dir = os.path.join(working_dir,"validation")
test_data_dir = os.path.join(working_dir,"test")
train_data_dir = os.path.join(working_dir,"train")

# Small Dataset Path for Testing Setup
# train_data_dir = os.path.join(working_dir,"training_small")


# Create new Directory where to save Model and Outputs later on
os.makedirs("model_outputs/" + model_name)


# Hyperparameters & Static Variables
epochs = 30
batch_size = 16
learning_rate = 0.001
input_shape = (img_width, img_height, color_channels)

# Learning Rate depending on epoch (Learning Rate Scheduler) with decaying evolvement
def lr_on_epoch(epoch_id, lr):
    new_lr = lr - (lr/20.0 * epoch_id)
    return new_lr

# Sequential CNN Model VGG3 Style
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape = input_shape))
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2))) # Default Parameters

model.add(Conv2D(64,(3,3)))
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(128,(3,3)))
model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(3))
model.add(Activation('softmax'))
model.summary()

# Compile Model - Save models per epoch and do Early Stopping with patience = 5
opt = RMSprop(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy' ,optimizer=opt,metrics=['accuracy'])
checkpoint = ModelCheckpoint(filepath='model_outputs/' + model_name + '/' +"model.{epoch:02d}-{val_accuracy:.2f}.h5", monitor="val_loss", mode="min", verbose=1, save_best_only=False, save_weights_only=False)
earlystop = EarlyStopping(monitor="val_accuracy", patience=5, mode="max", verbose=1, restore_best_weights=True)

lr_scheduler = LearningRateScheduler(lr_on_epoch, verbose=1)

logger = CSVLogger('model_outputs/' + model_name + '/' +'log.csv')


# Training and Validation Generator using Augmentation (comment out parameters not used in final setup)
train_data_generator = ImageDataGenerator(
    rescale=1./255,
    #width_shift_range=[-0.75,0.75],
    #height_shift_range=[-0.75,0.75],
    #fill_mode = 'constant', cval=190,
    horizontal_flip=True,
    #zoom_range = [0.5,1.5],
    brightness_range=[0.2,0.8],
    rotation_range=20
        )

validation_data_generator = ImageDataGenerator(
    rescale = 1./255
    )

# Generator Iteration for Training and Validation
# From Names of Subdirectories Keras derives the class labels
# It also counts the  number of items per subfolder
train_gen = train_data_generator.flow_from_directory(
    train_data_dir,
    target_size=(img_height,img_width),
    class_mode = 'categorical',
    shuffle = True
    )

# Use Augmentation also for Validation Generator - usually not done but yields better results
validation_gen_augm = train_data_generator.flow_from_directory(
    validate_data_dir,
    target_size=(img_height,img_width),
    class_mode = 'categorical',
        shuffle = False
    )

# Validation Iterator Version without Augmentation
validation_gen = validation_data_generator.flow_from_directory(
    validate_data_dir,
    target_size=(img_height,img_width),
    class_mode = 'categorical',
        shuffle = False
    )


# Model Training - use only a selection of defined Callbacks (LR Scheduler and Early Stopping)
history = model.fit_generator(
    generator = train_gen,
    steps_per_epoch = train_gen.samples // batch_size,
    epochs = epochs,
    validation_data = validation_gen_augm,
    validation_steps = validation_gen_augm.samples // batch_size,
    #callbacks = [earlystop]
    callbacks = [lr_scheduler, earlystop]
    )

# Save Final Model to Output Directory
model.save("model_outputs/" + model_name + '/' + model_name + '.npz')

# Function for Plotting diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label = 'train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    pyplot.legend(loc='best')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    pyplot.legend(loc='best')
    # Add Stuff
    pyplot.suptitle('Training History: Loss and Accuracy per Epoch')
    pyplot.tight_layout()
	# save plot to file
    pyplot.savefig('model_outputs/' + model_name + '/training_history.png')
    pyplot.close()

# Generate Plots and save them to the output directory
summarize_diagnostics(history)

# Plot Validation Confusion Matrix
pre_class = model.predict_generator(validation_gen)
labels = validation_gen.classes
pre_class_one_hot = pre_class.argmax(1)
cm_abs = confusion_matrix(labels, pre_class_one_hot)
cm_rel = confusion_matrix(labels, pre_class_one_hot, normalize='true')
print(cm_abs)
print(cm_rel)

class_names = validation_gen.class_indices
cm2_abs = ConfusionMatrixDisplay(cm_abs, display_labels=class_names)
cm2_rel = ConfusionMatrixDisplay(cm_rel, display_labels=class_names)
cm2_abs.plot()
cm2_rel.plot()
cm2_abs.figure_.savefig('model_outputs/' + model_name + '/cm_abs.png')
cm2_rel.figure_.savefig('model_outputs/' + model_name + '/cm_rel.png')

# Validation Score
score = model.evaluate_generator(validation_gen)
print(score)

# Accuracy
acc = accuracy_score(labels, pre_class_one_hot)

# Recall
rec = recall_score(labels, pre_class_one_hot, average=None)

# Precision
prec = precision_score(labels, pre_class_one_hot, average=None)






