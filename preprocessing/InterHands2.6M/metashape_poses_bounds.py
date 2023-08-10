import os
import Metashape
import argparse
import pdb
import numpy as np
from tqdm import tqdm
import cv2


METASHAPE_VERSION_REQUIRED = '1.5'

def check_compatibility():
    # Checking compatibility
    found_major_version = ".".join(Metashape.app.version.split('.')[:2])
    if found_major_version != METASHAPE_VERSION_REQUIRED:
        raise Exception(
            '\nWarning: Incompatible Metashape version! Found {}, required {}.'.format(found_major_version, METASHAPE_VERSION_REQUIRED)
        )

def gen_poses(scan_dir, data_dir, camera_file):    #TODO support factor
    print('Generating poses for {}'.format(scan_dir))
    doc = Metashape.Document()    
    # doc.open(os.path.join(scan_dir, METASHAPE_PROJECT_NAME+'.psx'))

    doc.addChunk()
    chunk = doc.chunks[0]

    chunk.importModel(os.path.join(scan_dir, 'mesh.obj'))
    # camera_folder = Path(data_dir).parent.absolute()
    # chunk.importCameras(os.path.join(camera_folder, 'camera.xml'))
    # print("cameras: ", chunk.cameras)

    #explicitly parsing camera params here cos the xml file is not being loaded successfully for some reason
    fs = cv2.FileStorage(camera_file, cv2.FILE_STORAGE_READ)

    for index in range(139):
    # for index in range(28):
        
        fn = fs.getNode("intrinsic-"+str(index))
        intrMat = fn.mat()
        # print("intr: ", intrMat)

        camera = chunk.addCamera()
        camera.label = str(index)
        camera.sensor = chunk.addSensor()
        camera.sensor.label = str(index)

        #intrinsics
        calibration = Metashape.Calibration()
        calibration.width 	= 334
        calibration.height	= 512
        calibration.cx 	= intrMat[0][2] - 167       #NOTE: values hardcoded for InterHand2.6M
        calibration.cy 	= intrMat[1][2] - 256
        calibration.f  	= intrMat[0][0]
        camera.sensor.user_calib = calibration
        camera.sensor.fixed = True
        #extrinsics
        fn = fs.getNode("extrinsic-"+str(index))
        extrMat = fn.mat()
        extrMat = np.vstack([extrMat, [0, 0, 0, 1]])
        # print("extr: ", extrMat)
        # camera.transform = Metashape.Matrix(np.linalg.inv(extrMat))
        camera.transform = Metashape.Matrix(extrMat)


    images_path = os.path.join(data_dir, 'images')
    poses = []

    for iter, camera in enumerate(tqdm(chunk.cameras)):

        #intrinsics
        f, w, h = camera.sensor.calibration.f, camera.sensor.calibration.width, camera.sensor.calibration.height
        # print(f, w, h)
        hwf = np.array([h,w,f]).reshape([3,1])


        #extrinsics
        c2w_mats = camera.transform # C2W
        c2w_mats = np.array(eval(str(c2w_mats)[len("Matrix("):-1])) # Metashape.Matrix to np array

        # CRS change from [x,-y,-z] (Metashape camera coord system) to [-y, x, z](NERF coord system) 
        # for rotation component
        c2w_mats[:, [1, 0]] = c2w_mats[:, [0, 1]] #swap x and -y
        c2w_mats[:, 2] = -1 * c2w_mats[:, 2]

        # 3x4 c2w affine transform, 3x1 intrinsics, 2 depth values, 1 image label 
        pose = np.concatenate([c2w_mats[:3], hwf], axis=1)


        #depth bounds
        depth = chunk.model.renderDepth(camera.transform, camera.sensor.calibration) #unscaled depth
        # depth.save(os.path.join(scan_dir,"metashape_render_"+str(iter)+".jpg"))
        depth = np.frombuffer(depth.tostring(), dtype=np.float32) 

        close_depth, inf_depth = np.min(depth[depth!=0]), np.max(depth) # non-zero min depth
        # print(close_depth)
        pose = np.concatenate([pose.ravel(), np.array([close_depth, inf_depth])], axis=0)
        poses.append(pose)

    poses = np.array(poses) # [N,17]
    n_imgfiles = len([f for f in os.listdir(images_path) if f.endswith('.png')])
    assert poses.shape[0] == n_imgfiles, 'Mismatch between imgs {} and poses {} !'.format(n_imgfiles, poses.shape[0])
    np.save(os.path.join(data_dir, 'poses_bounds.npy'), poses)
    np.savetxt(os.path.join(data_dir, 'poses_bounds.txt'), poses, fmt='%.3f')


def undistort(source_dir, scan_dir, data_dir):
    print('Undistorting masks and input for {}'.format(scan_dir))
    # Create new project
    doc = Metashape.Document()
    
    doc.open(os.path.join(scan_dir, METASHAPE_PROJECT_NAME+'.psx'))
    chunk = doc.chunks[0]
    chunk.importCameras(os.path.join(scan_dir, 'camera.xml'))
    chunk.importModel(os.path.join(scan_dir, '{}.obj'.format(METASHAPE_PROJECT_NAME)))

    camera = chunk.cameras[0]
    calibration = camera.sensor.calibration

    image = Metashape.Image(calibration.width, calibration.height, 'RGB')
    images_path = os.path.join(source_dir)

    for camera in tqdm(chunk.cameras):
        image_file = os.path.join(images_path, camera.label + '.png')
        if not os.path.isfile(image_file):
            print("Image {} not found".format(image_file))
            continue 
        if not camera.transform: #or not camera.type == Metashape.Camera.Type.Regular
            # os.remove(image_file)
            print("Warning!! skipping {} .. ".format(image_file))
            # continue
        else:
            image = image.open(os.path.join(images_path, camera.label + '.png'))
            image = image.undistort(calibration)
            image.save(os.path.join(data_dir, 'masks_undistorted', camera.label + '.png'))
          



if __name__ == '__main__':
    
    try:
        check_compatibility()
    except Exception as e:
        print(e)

    # Enable GPU usage
    Metashape.app.gpu_mask = 2** (len(Metashape.app.enumGPUDevices())) - 1

    arg_parser = argparse.ArgumentParser(
        description= "Prepare 3D scans for training a morphable head model with NeRF. "
    )

    arg_parser.add_argument(
        "--inputReconstructionsPath",
        "-i",
        dest="inputDir",
        required=True,
        help="path to reconstructions output by Agisoft. <path> containing <subjects>/<scan number>/scans",
    )
    arg_parser.add_argument(
        "--outputNeRFPath",
        "-o",
        dest="outputDir",
        required=True,
        help="path to write scans in NeRF-ready format. <path> will store <subjects>_<expression_id>",
    )

    arg_parser.add_argument(
        "--projectName",
        "-n",
        required=True,
        help="Project Name used in creating the reconstructions",
    )
    arg_parser.add_argument(
        "--imagePath",
        "-s",
        help="path of orignal images for unedited",
    )
    arg_parser.add_argument(
        "--cameraCalibFile",
        "-c",
        help="path of cameraCalibFile",
    )

    args = arg_parser.parse_args()
    root_path = args.inputDir
    # source_dir = args.imagePath
    nerf_dir = args.outputDir
    METASHAPE_PROJECT_NAME = args.projectName
    camera_file = args.cameraCalibFile

    gen_poses(root_path, nerf_dir, camera_file)
