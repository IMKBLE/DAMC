import time
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import torch
import numpy as np
import scipy.io as sio
from sklearn.cluster import KMeans
import models.metrics as metrics
import matplotlib.pyplot as plt 
# Load data
data_loader = CreateDataLoader(opt)
dataset_paired, paired_dataset_size = data_loader.load_data_pair()
#dataset_unpaired, unpaired_dataset_size = data_loader.load_data_unpair()
train_dataset_a = dataset_paired.dataset.train_data_a
train_dataset_b = dataset_paired.dataset.train_data_b
# untrain_dataset_a = dataset_unpaired.dataset.train_data_a
# untrain_dataset_b = dataset_unpaired.dataset.train_data_b
data_0 = sio.loadmat('rand1/label.mat')
data_dict=dict(data_0)
data0 = data_dict['label']
label_true = np.zeros((len(train_dataset_a)))
for i in range(len(train_dataset_a)):
    label_true[i]=data0[i]
# label_true_all = np.zeros(len(train_dataset_a)+2*len(untrain_dataset_a))
# for i in range(len(train_dataset_a)+2*len(untrain_dataset_a)):
#     label_true_all[i]=data0[i]
# label_true_UN = np.zeros(2*len(untrain_dataset_a))
# for i in range(2*len(untrain_dataset_a)):
#     label_true_UN[i]=data0[i+len(train_dataset_a)]
print(len(dataset_paired))
#print(len(dataset_unpaired))
print(len(train_dataset_b))
# print(len(untrain_dataset_a))
n_clusters = 5
n_com = 100
dim1 = 1750
dim2 = 79
# Create Model
model = create_model(opt)
visualizer = Visualizer(opt)

# Start Training
print('Start training')

#################################################
# Step1: Autoencoder
#################################################
print('step 1')
pre_epoch_AE = 30 # number of iteration for autoencoder pre-training
total_steps = 0
ACC_all=[]
NMI_all=[]
loss_ae = []
for epoch in range(1, pre_epoch_AE+1):
#    for i,(images_a, image_b) in enumerate(dataset_paired):

    for i in range(len(train_dataset_a)):

        images_a = torch.from_numpy(train_dataset_a[i]).view(1,1,dim1)
        images_b = torch.from_numpy(train_dataset_b[i]).view(1,1,dim2)
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - len(train_dataset_a) * (epoch - 1)
        model.set_input(images_a, images_b)
        model.optimize_parameters_pretrain_AE()
        loss_ae.append(model.loss_AE_pre.data.cpu())
        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors_AE_pre()
            visualizer.print_current_errors(epoch, epoch_iter, errors, iter_start_time)
#            if opt.display_id > 0:
#                visualizer.plot_current_errors(epoch, float(epoch_iter)/len(train_dataset_a), opt, errors)
    print('pretrain Autoencoder model (epoch %d, total_steps %d)' %
          (epoch, pre_epoch_AE)) 
    commonZ = []
    if epoch > 0:
        for i in range(len(train_dataset_a)):
            tempimage_a = torch.from_numpy(train_dataset_a[i]).view(1,1,dim1)
            tempimage_b = torch.from_numpy(train_dataset_b[i]).view(1,1,dim2)       
            model.set_input(tempimage_a, tempimage_b)
            t_200 =np.array(model.test_commonZ().data.view(n_com).tolist())
            commonZ.append(t_200)
        ##kmeans result
        estimator = KMeans(n_clusters)
        estimator.fit(commonZ)
        centroids =estimator.cluster_centers_
        label_pred = estimator.labels_
        acc = metrics.acc(label_true, label_pred)
        nmi = metrics.nmi(label_true, label_pred)
        ACC_all.append(acc)
        NMI_all.append(nmi)
        # Z_path = 'commonZ' + str(epoch)
        # sio.savemat(Z_path + '.mat', {'Z': commonZ})
        print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
                   % (acc, nmi))
        if acc > 0.95:
             break
        
centroids0 =estimator.cluster_centers_
########
center0 = torch.FloatTensor(centroids0).cuda()
model.clu.weights.data = center0
#########
sio.savemat('commonZAE.mat', {'Z':commonZ})
commonZ_step1 = commonZ

# comZ1 = []
# comZ2 = []
# for i in range(len(untrain_dataset_a)):
#     tempimage_a = torch.from_numpy(untrain_dataset_a[i]).view(1,1,dim1)
#     tempimage_b = torch.from_numpy(untrain_dataset_b[i]).view(1,1,dim2)
#     model.set_input(tempimage_a, tempimage_b)
#     dataset_fakeA, dataset_fakeB, t1_200, t2_200= model.test_unpaired()
#     t1_200 = t1_200.data.view(n_com).tolist()
#     t2_200 = t2_200.data.view(n_com).tolist()
#     comZ1.append(t1_200)
#     comZ2.append(t2_200)
# comZ12_ae = np.array(list(comZ1) + list(comZ2))
# commonZ_ae = np.array(list(commonZ) + list(comZ12_ae))
#
# estimator = KMeans(n_clusters)
# estimator.fit(commonZ_ae)
# centroids =estimator.cluster_centers_
# label_pred = estimator.labels_
# acc = metrics.acc(label_true_all, label_pred)
# nmi = metrics.nmi(label_true_all, label_pred)
# ACC_all.append(acc)
# NMI_all.append(nmi)
# print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
#       % (acc, nmi))
#
# estimator = KMeans(n_clusters)
# estimator.fit(comZ12_ae)
# centroids =estimator.cluster_centers_
# label_pred = estimator.labels_
# acc = metrics.acc(label_true_UN, label_pred)
# nmi = metrics.nmi(label_true_UN, label_pred)
# ACC_all.append(acc)
# NMI_all.append(nmi)
# print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
#       % (acc, nmi))
#
#
# fa_500 = []
# fb_500 = []
# for i in range(len(untrain_dataset_a)):
#     tempimage_a = torch.from_numpy(untrain_dataset_a[i]).view(1,1,dim1)
#     tempimage_b = torch.from_numpy(untrain_dataset_b[i]).view(1,1,dim2)
#     model.set_input(tempimage_a, tempimage_b)
#     dataset_fakeA, dataset_fakeB, t1_200, t2_200 = model.test_unpaired()
#     data_fakeA = dataset_fakeA.data.view(1,dim1).tolist()
#     data_fakeB = dataset_fakeB.data.view(1,dim2).tolist()
#     fa_500.append(data_fakeA)
#     fb_500.append(data_fakeB)
#
# test_dataset_A2000 = np.array(list(train_dataset_a) + list(untrain_dataset_a) + list(fa_500))
# test_dataset_B2000 = np.array(list(train_dataset_b) + list(fb_500) + list(untrain_dataset_b))
# sio.savemat('fakea1.mat',{'fa':fa_500})
# sio.savemat('fakeb1.mat',{'fb':fb_500})
#
# commonZ_step1 = []
# for i in range(len(test_dataset_A2000)):
#     tempimage_a = torch.from_numpy(test_dataset_A2000[i]).view(1,1,dim1)
#     tempimage_b = torch.from_numpy(test_dataset_B2000[i]).view(1,1,dim2)
#     model.set_input(tempimage_a, tempimage_b)
#     t_200 =np.array(model.test_commonZ().data.view(n_com).tolist())
#     commonZ_step1.append(t_200)
# estimator = KMeans(n_clusters)
# estimator.fit(commonZ_step1)
# centroids =estimator.cluster_centers_
# label_pred = estimator.labels_
# acc = metrics.acc(label_true_all, label_pred)
# nmi = metrics.nmi(label_true_all, label_pred)
# ACC_all.append(acc)
# NMI_all.append(nmi)
# print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
#       % (acc, nmi))
# #sio.savemat('ZAE.mat', {'commonZ_AE':commonZ_step1})
#
#

 #################################################
 # Step2: CycleGAN
 #################################################
loss_ae_g=[]
loss_g_g=[]
loss_da_g=[]
loss_db_g=[]
print('step 2')
pre_epoch_cycle = 5# number of iteration for CycleGAN training
total_steps = 0
for epoch in range(1, pre_epoch_cycle+1):

    epoch_start_time = time.time()
     #for i,(images_a, images_b) in enumerate(dataset_unpaired):
    for i in range(len(train_dataset_a)):
        images_a_2 = torch.from_numpy(train_dataset_a[i]).view(1,1,dim1)
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - paired_dataset_size * (epoch - 1)
        images_b_2 = torch.from_numpy(train_dataset_b[i]).view(1,1,dim2)
        model.set_input(images_a_2, images_b_2)
        model.optimize_parameters_pretrain_cycleGAN()

        loss_ae_g.append(model.loss_ae.data.cpu())
        loss_g_g.append(model.loss_GAB.data.cpu())
        loss_da_g.append(model.loss_D_A.data.cpu())
        loss_db_g.append(model.loss_D_B.data.cpu())
        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors_cycle()
            visualizer.print_current_errors(epoch, epoch_iter, errors, iter_start_time)
 #            if opt.display_id > 0:
 #                visualizer.plot_current_errors(epoch, float(epoch_iter)/unpaired_dataset_size, opt, errors)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, pre_epoch_cycle, time.time() - epoch_start_time))





    commonZ = []
    for i in range(len(train_dataset_a)):
        tempimage_a = torch.from_numpy(train_dataset_a[i]).view(1,1,dim1)
        tempimage_b = torch.from_numpy(train_dataset_b[i]).view(1,1,dim2)
        model.set_input(tempimage_a, tempimage_b)
        t_200 =np.array(model.test_commonZ().data.view(n_com).tolist())
        commonZ.append(t_200)
    estimator = KMeans(n_clusters)
    estimator.fit(commonZ)
    centroids_step2 =estimator.cluster_centers_
    label_pred = estimator.labels_
    acc = metrics.acc(label_true, label_pred)
    nmi = metrics.nmi(label_true, label_pred)
    ACC_all.append(acc)
    NMI_all.append(nmi)
    print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
      % (acc, nmi))
     
sio.savemat('commonZGAN.mat', {'Z':commonZ})
commonZ_step2 = commonZ
 #    if epoch > pre_epoch_cycle/2:
 #        model.update_learning_rate()
 #    commonZ = []
 #    if epoch > 0:
 #        for i in range(len(train_dataset_a)):
 #            tempimage_a = torch.from_numpy(train_dataset_a[i]).view(1,1,dim1)
 #            tempimage_b = torch.from_numpy(train_dataset_b[i]).view(1,1,dim2)
 #            model.set_input(tempimage_a, tempimage_b)
 #            t_200 =np.array(model.test_commonZ().data.view(n_com).tolist())
 #            commonZ.append(t_200)
 #        ##kmeans result
 #        estimator = KMeans(n_clusters)
 #        estimator.fit(commonZ)
 #        centroids =estimator.cluster_centers_
 #        label_pred = estimator.labels_
 #        acc = metrics.acc(label_true, label_pred)
 #        nmi = metrics.nmi(label_true, label_pred)
 #        ACC_all.append(acc)
 #        NMI_all.append(nmi)
 #        print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
 #                   % (acc, nmi))


q1 = 1.0 / (1.0 + (torch.sum(torch.pow(torch.unsqueeze(torch.FloatTensor(commonZ_step1), 1)-torch.FloatTensor(centroids0), 2), 2) ))
q = torch.t(torch.t(q1) / torch.sum(q1, 1))
p1 = torch.pow(q,2)/torch.sum(q,0)
p = torch.t(torch.t(p1)/torch.sum(p1,1))
 #center = torch.FloatTensor(centroids).cuda()
 #center = torch.FloatTensor(centroids_step2).cuda()
 #model.clu.weights.data = center


 #################################################
 # Step3:  VIGAN
 #################################################
print('step 3')
total_steps = 0
 #eee = []
 #ACC_all=[]
 #NMI_all=[]
loss_ave = []
loss_temp = torch.zeros(1)
for epoch in range(1, opt.niter + opt.niter_decay + 1):
    if epoch>15:
        break
    epoch_start_time = time.time()
 	# You can use paired and unpaired data to train. Here we only use paired samples to train.
     #for i,(images_a, images_b) in enumerate(dataset_paired):
    q = []
    for i in range(len(train_dataset_a)):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - paired_dataset_size * (epoch - 1)
         ##no clustering_loss
 #        model.set_input(images_a, images_b)
 #        model.optimize_parameters()
 #        t_200 =np.array(model.test_commonZ().data.view(200).tolist())
 #        commonZ.append(t_200)

         ##add clustering_loss
        images_a = torch.from_numpy(train_dataset_a[i]).view(1,1,dim1)
        images_b = torch.from_numpy(train_dataset_b[i]).view(1,1,dim2)
        pp_i = p[i].cuda()
        model.set_input_train(images_a,images_b,pp_i)
        model.optimize_AECL()
        q_i = model.q.data
        qi = q_i.view(n_clusters ).tolist()
        q.append(qi)
        loss_temp = loss_temp + model.loss_AE_CL.data.cpu()


    if total_steps % opt.print_freq == 0:
        errors = model.get_current_errors_AE_CL()
        visualizer.print_current_errors(epoch, epoch_iter, errors, iter_start_time)
        if opt.display_id > 0:
            visualizer.plot_current_errors(epoch, float(epoch_iter)/paired_dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    loss_ave.append((loss_temp/len(train_dataset_a)).tolist())
    loss_temp = torch.zeros(1)
     ##kmeans result
    commonZ = []
    for i in range(len(train_dataset_a)):
        tempimage_a = torch.from_numpy(train_dataset_a[i]).view(1,1,dim1)
        tempimage_b = torch.from_numpy(train_dataset_b[i]).view(1,1,dim2)
        model.set_input(tempimage_a, tempimage_b)
        t_200 =np.array(model.test_commonZ().data.view(n_com).tolist())
        commonZ.append(t_200)
     ##kmeans result
    estimator = KMeans(n_clusters)
    estimator.fit(commonZ)
    centroids =estimator.cluster_centers_
    label_pred = estimator.labels_
    acc = metrics.acc(label_true, label_pred)
    nmi = metrics.nmi(label_true, label_pred)
    ACC_all.append(acc)
    NMI_all.append(nmi)
    sio.savemat('acc.mat', {'ACC_all':ACC_all})
    sio.savemat('nmi.mat', {'NMI_all':NMI_all})
    sio.savemat('loss.mat', {'loss_all':loss_ave})
    Z_path = 'commonZ'+ str(epoch)
    sio.savemat(Z_path+'.mat',{'Z':commonZ})
    print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
          % (acc, nmi))


 #    loss_ave.append((loss_temp/len(train_dataset_a)).tolist())
    q = torch.FloatTensor(q)
    p1 = torch.pow(q,2)/torch.sum(q,0)
    p = torch.t(torch.t(p1)/torch.sum(p1,1))


    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
         #Z_path = 'commonZ'+ str(epoch)
         #sio.savemat(Z_path+'.mat',{'Z':commonZ})

        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()
 #sio.savemat('error.mat', {'error':eee})

x=torch.linspace(1, len(ACC_all), steps=len(ACC_all))
x = x.numpy()
y_acc = torch.FloatTensor(ACC_all).numpy()
y_nmi = torch.FloatTensor(NMI_all).numpy()
y_loss = torch.FloatTensor(loss_ave).numpy()
plt.cla()
plt.plot(x, y_nmi, c='red', label='nmi')
plt.plot(x, y_acc, c='blue', label='acc')
 #plt.plot(x, y_loss, c='green', label='loss')


