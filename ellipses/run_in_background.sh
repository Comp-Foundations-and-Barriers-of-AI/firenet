#! /usr/bin/bash

#nohup nice -n 19 /opt/uio/modules/rhel8/easybuild/software/Python/3.7.4-GCCcore-8.3.0/bin/python3 -u Demo_test_lasso_stability_auto.py &> out/out1.txt &

#nohup nice -n 1 /opt/uio/modules/rhel8/easybuild/software/Python/3.7.4-GCCcore-8.3.0/bin/python3 -u data_management.py &> out/out9.txt &

nohup nice -n 10 /opt/uio/modules/rhel8/easybuild/software/Python/3.7.4-GCCcore-8.3.0/bin/python3 -u Demo_train_nn.py &> out/out3.txt &

#nohup nice -n 19 /opt/uio/modules/rhel8/easybuild/software/Python/3.7.4-GCCcore-8.3.0/bin/python3 -u Demo_test_lasso_stability_images.py &> out/out3.txt &


#nohup nice -n 19 /opt/uio/modules/rhel8/easybuild/software/Python/3.7.4-GCCcore-8.3.0/bin/python3 -u Demo_test_automap_stability.py &> out/out3.txt &


