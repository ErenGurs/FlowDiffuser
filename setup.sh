

#mkdir ckpts/
#wget -P ckpts/ https://github.com/DQiaole/MemFlow/releases/download/v1.0.0/MemFlowNet_sintel.pth
#wget -P ckpts/ https://github.com/DQiaole/MemFlow/releases/download/v1.0.0/MemFlowNet_things.pth

alias conductor='AWS_EC2_METADATA_DISABLED=true aws --endpoint-url https://conductor.data.apple.com'

if [ ! -d  "datasets/Sintel" ]; then
    mkdir -p "datasets"
    conductor s3 cp s3://egurses-diffusion/Datasets/MPI-Sintel-complete.zip  ./
    echo "Extracting MPI-Sintel-complete.zip ..."
    unzip MPI-Sintel-complete.zip -d datasets > /dev/null 2>&1
    mv datasets/MPI-Sintel-complete datasets/Sintel
fi

# conductor s3 cp --recursive s3://egurses-frc/ImagePairs/ ./ImagePairs/
#conductor s3 cp s3://mingchen_li/real_validation1.zip ./
# s3zip -e s3://mingchen_li/real_validation1.zip ./test_real/

if [[ $(conda env list) != *"flowdiffuser"* ]]; then
    conda create --name flowdiffuser --clone iris
    conda activate flowdiffuser

    pip install timm #timm=0.4.12 # imageio matplotlib tensorboard scipy opencv-python h5py tqdm
fi

# To be removed later:
# An experiment to run MemFlow on sequence data
#conductor s3 cp --recursive s3://egurses-frc/BMS/input/bms.video.new/ ./bms.video.new/
#mkdir val5_holes_new/
#ffmpeg -i bms.video.new/715G3Q7O2IS9C8NJ_IMG_0073.MOV  val5_holes_new/'%04d.png'
#inference.py --name MemFlowNet --stage sintel --restore_ckpt ckpts/MemFlowNet_things.pth --seq_dir ./val5_holes_new/ --vis_dir temp/val5_holes_new/
