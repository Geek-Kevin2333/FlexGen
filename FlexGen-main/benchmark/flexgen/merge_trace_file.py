import os
import json
import argparse
import shutil

# deprecated
node_ip_lists = []


# deprecated
def download_trace_logs(args, profix, postfix, ips=node_ip_lists):
    if os.path.isdir('./' + profix):
        # os.rmdir('./'+prefix)
        shutil.rmtree('./' + profix)
    os.mkdir('./' + profix)
    for i in range(args.world_size):
        os.system("scp -i ../binhang_ds3_aws_oregon.pem ubuntu@" + ips[i] +
                  ":~/GPT-home-private/trace_json/" + profix + '_' + str(i) + postfix + ' ./' + profix)


def merge_logs(args):
    result = []
    # The function initializes an empty list result and a reference to an initial minimum timestamp value current_min_stamp as positive infinity.
    current_min_stamp = float('inf')
    for i in range(args.world_size):
        # It then iterates through each worker node's log file using a for loop that goes from 0 to args.world_size.
        print(i)
        trace_file = "prefilling/" + "generate_overlap_" + str(args.overlap) + "num_gpu_batches_" + str(
            args.num_gpu_batches) + "_percent_" + str(
            args.percent) + "pp_rank:" + str(i) + \
                     '.json'
        with open(trace_file) \
                as inputJson:
            current_trace = json.load(inputJson)
            inputJson.close()
            if i == 0:
                for log in current_trace:
                    current_min_stamp = min(log['ts'], current_min_stamp)
            for log in current_trace:
                log['pid'] = args.model + ' node ' + str(i)
                log['ts'] = log['ts'] - current_min_stamp
            result.extend(current_trace)
    print(len(result))
    file_name="/trace_json/" +'dist/'+ args.model +'_numgpubatches_'+args.num_gpu_batches+'_gpubatchsize_'+args.gpu_batch_size+"_comm_device_"+args.comm_device +'_percent_' + str(args.percent)+\
              "_compress_weight"+args.compress_weight+"_compress_cache"+args.compress_cache + '.json'
    dir_name = file_name[0:file_name.rindex('/')]
    print(dir_name)
    os.makedirs(dir_name, exist_ok=True)
    with open(file_name, 'w') as outputJson:
        json.dump(result, outputJson)


def main():
    parser = argparse.ArgumentParser(description='OPT')
   # parser.add_argument('--world-size', type=int, default=12, metavar='N', help='distributed cluster size (default: 3)')
    parser.add_argument('--world-size', type=int, default=4, metavar='N',
                        help='distributed cluster size (default: 4)')
    parser.add_argument('--model', type=str, default='opt-6.7b', metavar='S',
                        help='use which model: opt-6.7b or opt-30b opt-175b.')
    parser.add_argument('--profix', type=str, default='opt', metavar='S',
                        help='postfix of the tracing file name.')
    parser.add_argument('--postfix', type=str, default='tidy_profiling_real', metavar='S',
                        help='postfix of the tracing file name.')
    parser.add_argument('--num-gpu-batches', type=str, default='1', metavar='S',
                        help='num-gpu-batches')
    parser.add_argument('--gpu-batch-size', type=str, default='24', metavar='S',
                        help='gpu-batch-size.')
    parser.add_argument('--overlap', type=str, default='false', metavar='S',
                        help='overlap')
    parser.add_argument('--comm-device', type=str, default='cpu', metavar='S',
                        help='comm-device')
    parser.add_argument('--path', type=str, default='_DUMMY_', metavar='S',
                        help='comm-device')
    parser.add_argument('--percent', nargs="+", type=int,
                        default=[100, 0, 100, 0, 100, 0], metavar='S',
                        help='percent')
    parser.add_argument('--cut-gen-len', type=str, default='2', metavar='S',
                        help='cut-gen-len')
    parser.add_argument('--compress-weight', type=str, default='False', metavar='S',
                        help='compress-weight')
    parser.add_argument('--compress-cache', type=str, default='False', metavar='S',
                        help='compress-weight')
    args = parser.parse_args()
    merge_logs(args)


if __name__ == '__main__':
    main()