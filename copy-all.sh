for pod in `kubectl get pods -o=name | grep raycluster | sed "s/^.\{4\}//"`
do
    kubectl cp main $pod:/home/ray
    kubectl cp price-trace $pod:/home/ray
done
