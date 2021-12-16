import torch
import copy
import parameter as p

def pretrain_print(epoch, batch_idx, data, train_loader, loss, num_example, correct):
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                   batch_idx * len(data), 
                                                                   len(train_loader.dataset),
                                                                   100. * batch_idx/len(train_loader), loss.item()))
    
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\ttrain_accuracy: {:.6f}'.format(epoch,
                                                                   batch_idx * len(data), 
                                                                   len(train_loader.dataset),
                                                                   100. * batch_idx/len(train_loader), correct.item()/num_example))
    
    
    

def test_print(test_loss, correct, instance_counter):
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, 
                                                                              instance_counter,
                                                                              100. * correct / (instance_counter)))
    
    
    
def posttrain_print(epoch, idx, class_index, class_accuracy, decoder, reg, network, data, target, foldername, unique_class):
    
    with torch.no_grad():
        sum_params = 0
        class_images, decoder_outputs, network_latents = {}, {}, {}

        if idx % 100 == 0:
            print('#'*20)
            for layer in range(len(network.layers)):
                print('epoch: {} , batch_count: {} , class_index: {} , layer {} , non_zeros {}'.format(epoch, 
                                                                                                       idx, 
                                                                                                       class_index, 
                                                                                                       layer+1, 
                                                                                                       helper.count_nonzero(network)[layer]))

            print("epoch: {}, batch_count: {}, cluster_centers: {} ".format(epoch,idx,reg.means))
            print("class_index: {}     accuracy:{}".format(class_index,class_accuracy))
            network2 = copy.deepcopy(network).cuda()
            network2.eval()
            for eval_class in unique_class:
                decoder_output, decoder_input, network_latent = helper.generate_decoded_image(decoder,network2, data.cuda(),target.cuda(), eval_class)
                class_images[eval_class] = decoder_input
                decoder_outputs[eval_class] = decoder_output
                network_latents[eval_class] = network_latent

            print("*"*20,"activation_check","*"*20)
            print("latent_output_check: {}     decoder_output_check: {} ".format(helper.check_activations(network_latents),helper.check_activations(decoder_outputs)))

            helper.plot_class_image(class_images,"class",foldername,epoch,idx,class_index)
            helper.plot_class_image(decoder_outputs,"decoder_output",foldername,epoch,idx,class_index)