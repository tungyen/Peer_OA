import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
import cv2
import os
from ultralytics import YOLO

ckpts_path = "ckpts"
if not os.path.exists(ckpts_path):
    os.mkdir(ckpts_path)
class ZED2YOLOProcessor(Node):
    def __init__(self):
        super().__init__('zed2_yolo_processor')
        
        self.image_sub = self.create_subscription(
            Image,
            '/robot1/zed2i/left/image_rect_color',
            self.image_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/robot1/zed2i/left/camera_info',
            self.camera_info_callback,
            10
        )
        self.bridge = CvBridge()
        
        ptName_seg = os.path.join(ckpts_path, "yolov11_detection.pt")
        ptName_detect = os.path.join(ckpts_path, "yolov11_seg.pt")
        self.object_detection_model = YOLO(ptName_detect)
        self.segmentation_model = YOLO(ptName_seg)
    
        self.get_logger().info("ZED2 YOLO Processor initialized.")

    def camera_info_callback(self, msg):
        self.get_logger().info(f"Received CameraInfo: width={msg.width}, height={msg.height}")

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if cv_image.shape[2] == 4:
            cv_image = cv_image[:, :, :3]
        
        detection_results = self.object_detection_model(cv_image, save=True, conf=0.5, project="zed_detect_res", name="")
        self.get_logger().info(f"Object detection results: {detection_results}")
        
        segmentation_results = self.segmentation_model(cv_image, save=True, conf=0.5, project="zed_seg_res", name="")
        self.get_logger().info(f"Segmentation results: {segmentation_results}")
        
        # annotated_frame = detection_results[0].plot()
        # cv2.imshow('YOLO Detection', annotated_frame)
        # cv2.waitKey(1)

        # segmented_frame = segmentation_results[0].masks.data if segmentation_results[0].masks else None
        # if segmented_frame is not None:
        #     self.get_logger().info("Segmentation masks generated.")

def main(args=None):
    rclpy.init(args=args)
    node = ZED2YOLOProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
