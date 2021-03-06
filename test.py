import numpy as np
import torch

from utils import io_utils
from data import generator
from torch.autograd import Variable


def test_one_shot(args, model, test_samples=5000, partition='test'):
    io = io_utils.IOStream('checkpoints/' + args.exp_name + '/run.log')

    io.cprint('\n**** TESTING WITH %s ***' % (partition,))
    loader = generator.Generator(args.dataset_root, args, partition=partition, dataset=args.dataset)

    [enc_nn, metric_nn, softmax_module] = model
    enc_nn.eval()
    metric_nn.eval()
    correct = 0
    total = 0
    iterations = int(test_samples / args.batch_size_test)
    accuracy_array = np.zeros(1)
    # print('accuracy array before')
    # print(accuracy_array)
    for i in range(iterations):
        data = loader.get_task_batch(batch_size=args.batch_size_test, n_way=args.test_N_way,
                                     num_shots=args.test_N_shots, unlabeled_extra=args.unlabeled_extra)
        [x, labels_x_cpu, _, _, xi_s, labels_yi_cpu, oracles_yi, hidden_labels] = data

        if args.cuda:
            xi_s = [batch_xi.cuda() for batch_xi in xi_s]
            labels_yi = [label_yi.cuda() for label_yi in labels_yi_cpu]
            oracles_yi = [oracle_yi.cuda() for oracle_yi in oracles_yi]
            hidden_labels = hidden_labels.cuda()
            x = x.cuda()
        else:
            labels_yi = labels_yi_cpu

        xi_s = [Variable(batch_xi) for batch_xi in xi_s]
        labels_yi = [Variable(label_yi) for label_yi in labels_yi]
        oracles_yi = [Variable(oracle_yi) for oracle_yi in oracles_yi]
        hidden_labels = Variable(hidden_labels)
        x = Variable(x)

        # Compute embedding from x and xi_s
        z = enc_nn(x)[-1]
        zi_s = [enc_nn(batch_xi)[-1] for batch_xi in xi_s]
        dirichlet_flag = True
        # Compute metric from embeddings
        output, out_logits, x_next, W_for_dirichle = metric_nn(inputs=[z, zi_s, labels_yi, oracles_yi, hidden_labels, dirichlet_flag])
        output = out_logits
        y_pred = softmax_module.forward(output)
        y_pred = y_pred.data.cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        labels_x_cpu = labels_x_cpu.numpy()
        labels_x_cpu = np.argmax(labels_x_cpu, axis=1)

        for row_i in range(y_pred.shape[0]):
            if y_pred[row_i] == labels_x_cpu[row_i]:
                correct += 1
            total += 1

        if (i + 1) % 100 == 0:
            io.cprint('{} correct from {} \tAccuracy: {:.3f}%)'.format(correct, total, 100.0 * correct / total))
            #print('the accuracy?')
            #print(100.0*correct/total)
            accuracy_array = np.append(accuracy_array, 100.0 * correct / total)  #print(accuracy_array)

    #calculate dirichle energy

    dirichle_energy = 0;
    #print("size of x", x_next.size())
    #print("size of x column", x_next[0,1,:].size())
    #print("size of w", W_for_dirichle.size())
    #print("size of w[i,j]", W_for_dirichle[0,1,1,0].size())
    #x_next.cpu()
    #W_for_dirichle.cpu()

    for line1 in range(6):
        for line2 in range(6):
            #print("x1:",x_next[:,line1,:].size())
            #print("x2:", x_next[:,line2,:].size())
            #print("sub:",(x_next[:,line1,:] - x_next[:,line2,:]).size())
            #print("pow vec", (torch.pow((x_next[:,line1,:] - x_next[:,line2,:]),2)).size())
            #print("sum:",(torch.sum(torch.pow((x_next[:,line1,:] - x_next[:,line2,:]),2),1)).size())
            #print("aij", W_for_dirichle[:,line1,line2,1])
            #print("before mul: ", torch.sum(torch.pow((x_next[:,line1,:] - x_next[:,line2,:]),2),1))
            #print("after mul: ", W_for_dirichle[:,line1,line2,1]*(torch.sum(torch.pow((x_next[:,line1,:] - x_next[:,line2,:]),2),1)))
            calc = W_for_dirichle[:, line1, line2, 1] * (
                torch.sum(torch.pow((x_next[:, line1, :] - x_next[:, line2, :]), 2), 1))
            #print("calc:", calc)
            dirichle_energy = dirichle_energy + calc  #print("dirichler in test:", dirichle_energy)

    #print("type of dirichle: ", type(dirichle_energy))
    #print("type after casting: ", type(int(dirichle_energy)))
    io.cprint('{} correct from {} \tAccuracy: {:.3f}%)'.format(correct, total, 100.0 * correct / total))
    io.cprint('*** TEST FINISHED ***\n'.format(correct, total, 100.0 * correct / total))
    #np.append(accuracy_array, 100.0*correct/total)

    enc_nn.train()
    metric_nn.train()

    return 100.0 * correct / total, np.mean(accuracy_array[1:]), float(torch.mean(dirichle_energy))