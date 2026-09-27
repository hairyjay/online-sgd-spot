# for pod in `kubectl get pods -o=name | grep raycluster | sed "s/^.\{4\}//"`
# do
#     kubectl exec $pod -- mkdir -p spot_aws/data/ImageNet
#     kubectl exec $pod -- curl -L -o spot_aws/data/ImageNet/dataset.zip --header "Authorization: Bearer $(jq -r .key ~/.kaggle/kaggle.json)" "https://www.kaggle.com/api/v1/datasets/download/mayurmadnani/imagenet-dataset" &
# done

# wait

for pod in `kubectl get pods -o=name | grep raycluster | sed "s/^.\{4\}//"`
do
    kubectl exec $pod -- sudo apt-get install unzip
    kubectl exec $pod -- unzip spot_aws/data/ImageNet/dataset.zip &
done

wait

for pod in `kubectl get pods -o=name | grep raycluster | sed "s/^.\{4\}//"`
do
    kubectl exec $pod -- rm -rf spot_aws/data/ImageNet/train
    kubectl exec $pod -- rm -rf spot_aws/data/ImageNet/test
    kubectl exec $pod -- rm -rf spot_aws/data/ImageNet/val
    kubectl exec $pod -- mv train spot_aws/data/ImageNet/train
    kubectl exec $pod -- mv test spot_aws/data/ImageNet/test
    kubectl exec $pod -- mv val spot_aws/data/ImageNet/val
done

# kubectl exec $(kubectl get pods -l key=value -o name) -- curl -L -o spot_aws/data/ImageNet/dataset.zip --header "Authorization: Bearer $(jq -r .key ~/.kaggle/kaggle.json)" "https://www.kaggle.com/api/v1/datasets/download/mayurmadnani/imagenet-dataset"
