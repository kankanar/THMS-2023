from densedilatedunetmodel import *
from data import *
import tensorflow as tf
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/hgr/train','image','label',data_gen_args,save_to_dir = None)

model = dense_unet()
model_checkpoint = ModelCheckpoint('unet_hgr.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=1500,epochs=100,callbacks=[model_checkpoint])

testGene = testGenerator("data/hgr/test/image")
results = model.predict_generator(testGene,1000,verbose=1)
saveResult("data/hgr/test/result",results)
