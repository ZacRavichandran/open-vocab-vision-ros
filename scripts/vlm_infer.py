import rospy

from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from sensor_msgs.msg import Image

from open_vocab_vision_ros.detection_node import decode_img_msg
from open_vocab_vision_ros.srv import Query, QueryRequest, QueryResponse

from open_vocab_vision_ros.vlm.vlm import VLMWrapper


class VLMInfer:
    def __init__(self) -> None:
        model = rospy.get_param("~model", "llava-hf/vip-llava-7b-hf")
        input_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        classes_list = rospy.get_param(
            "~classes", "parking lot, road, sidewalk, park, other"
        )

        self.vlm = VLMWrapper(model=model, classes=classes_list)

        self.latest_img = None
        self.img_sub = rospy.Subscriber(input_topic, Image, self.img_cbk)
        self.cls_scene = rospy.Service("~classify_scene", Trigger, self.classify_scene)
        self.open_cls_scene = rospy.Service(
            "~open_classify_scene", Trigger, self.open_classify_scene
        )
        self.srv = rospy.Service("~query_scene", Query, self.answer_scene)

    def img_cbk(self, img: Image) -> None:
        self.latest_img = decode_img_msg(img)

    def classify_scene(self, req: TriggerRequest) -> TriggerResponse:
        if self.latest_img is None:
            msg = "unknown"
            return TriggerResponse(success=True, message=msg)

        msg, class_id = self.vlm.classify_scene(image=self.latest_img)
        rospy.loginfo(f"vlm returned {msg}")

        return TriggerResponse(success=True, message=class_id)

    def open_classify_scene(self, req: TriggerRequest) -> TriggerResponse:
        if self.latest_img is None:
            msg = "unknown"
            return TriggerResponse(success=True, message=msg)

        msg, answer = self.vlm.open_classify_scene(image=self.latest_img)
        rospy.loginfo(f"vlm returned {msg}")

        return TriggerResponse(success=True, message=answer)

    def answer_scene(self, req: QueryRequest) -> TriggerResponse:
        if self.latest_img is None:
            msg = "unknown"
            return TriggerResponse(success=False, message=msg)

        msg = self.vlm.open_query(prompt=req.query, image=self.latest_img)
        parsed = msg.split(req.query)[-1]
        rospy.loginfo(f"vlm returned {msg}")

        return TriggerResponse(success=True, message=parsed)


if __name__ == "__main__":
    rospy.init_node("vlm_node")

    node = VLMInfer()

    rospy.spin()
