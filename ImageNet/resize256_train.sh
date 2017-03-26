for name in /home/gpu-admin/data/ImageNet/train/*/*.JPEG; 
do
( convert -resize 256x256\! $name $name ) &
    #echo $name
done
wait