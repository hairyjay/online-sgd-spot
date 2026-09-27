kubectl cp raycluster-head-8lnz4:/home/ray/spot_aws/data/ImageNet/train ./ImageNet/train
kubectl cp raycluster-head-8lnz4:/home/ray/spot_aws/data/ImageNet/test ./ImageNet/test
kubectl cp raycluster-head-8lnz4:/home/ray/spot_aws/data/ImageNet/val ./ImageNet/val

pod="raycluster-worker-doll-w-44r6h"
kubectl exec $pod -- mkdir -p spot_aws/data/ImageNet
kubectl cp ./ImageNet $pod:/home/ray/spot_aws/data/ImageNet/ &
# for pod in `kubectl get pods -o=name | grep raycluster-worker | sed "s/^.\{4\}//"`
# do
#     kubectl exec $pod -- mkdir -p spot_aws/data/ImageNet
#     kubectl cp ./ImageNet $pod:/home/ray/spot_aws/data/ImageNet/ &
# done

# wait

# kubectl exec $(kubectl get pods -l key=value -o name) -- curl -L -o spot_aws/data/ImageNet/dataset.zip --header "Authorization: Bearer $(jq -r .key ~/.kaggle/kaggle.json)" "https://www.kaggle.com/api/v1/datasets/download/mayurmadnani/imagenet-dataset"
