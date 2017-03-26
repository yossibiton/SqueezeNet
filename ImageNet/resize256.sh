for name in /home/gpu-admin/data/ImageNet/val/*.JPEG; do
    convert -resize 256x256\! $name $name
done