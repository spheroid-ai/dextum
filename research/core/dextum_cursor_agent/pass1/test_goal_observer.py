"""Interface-level checks; stdlib only, no simulator."""
import math
import unittest
from .goal_observer import GoalObserver, PoseObservation, multiply


def pose(tick,yaw=30.,tilt=0.,source='pose_estimator',z=0.):
    y=math.radians(yaw)/2;t=math.radians(tilt)/2
    q=multiply((math.cos(y),0.,0.,math.sin(y)),(math.cos(t),math.sin(t),0.,0.))
    return PoseObservation(tick,q,z,source)


class ObserverTests(unittest.TestCase):
    def observer(self,blackout=True):
        o=GoalObserver(30,blackout=blackout)
        self.assertTrue(o.issue_at_contact(pose(0,0,source='camera')))
        return o

    def test_hold_and_latched_reporting(self):
        o=self.observer()
        for t in range(39):self.assertFalse(o.update(t,pose(t))['goal_now'])
        self.assertTrue(o.update(39,pose(39))['goal_now'])
        result=o.update(40,pose(40,yaw=80))
        self.assertFalse(result['goal_now']);self.assertTrue(result['ever_observed'])

    def test_tilt_and_drop_reject_yaw_only_false_success(self):
        for tilt,z in [(31.,0.),(0.,-.051)]:
            o=self.observer()
            for t in range(45):r=o.update(t,pose(t,tilt=tilt,z=z))
            self.assertFalse(r['ever_observed'])

    def test_blackout_and_missing_pose_do_not_use_camera(self):
        o=self.observer()
        for t in range(45):self.assertFalse(o.update(t,pose(t,source='camera'))['known'])
        self.assertFalse(o.ever_observed)
        self.assertFalse(o.update(45,None)['known'])
        with self.assertRaises(ValueError):PoseObservation(0,(1,0,0,0),0.,'evaluation_truth')

    def test_consecutive_ticks_and_quaternion_sign(self):
        o=self.observer(blackout=False)
        for t in range(20):o.update(t,pose(t,source='camera'))
        p=pose(21,source='camera')
        q=PoseObservation(21,tuple(-x for x in p.quaternion_wxyz),0.,'camera')
        self.assertEqual(o.update(21,q)['hold_steps'],1)
        with self.assertRaises(ValueError):o.update(21,q)

    def test_task_loop_uses_pose_and_keeps_success_latched(self):
        from .runtime import Packet, TaskLoop, reorientation_dag
        loop=TaskLoop(reorientation_dag('reorient by 30 degrees'),blackout=False)
        for t in range(43):
            measurement=pose(t,yaw=0 if t==0 else (30 if t<42 else 80),source='camera')
            packet=Packet(t,(1.,)*11,(0.,)*11,(False,)*11,(0.,)*7,
                          camera_yaw_deg=0.,camera_shape='cube',camera_valid=True,pose=measurement)
            choice=loop.update(packet)
        self.assertTrue(loop.goal_observed)
        self.assertEqual(choice['node'],'N6')

    def test_legacy_face_angle_cannot_claim_aligned_success(self):
        from .runtime import Packet, TaskLoop, reorientation_dag
        loop=TaskLoop(reorientation_dag('reorient by 30 degrees'),blackout=False)
        for t in range(45):
            loop.update(Packet(t,(1.,)*11,(0.,)*11,(False,)*11,(0.,)*7,
                        camera_yaw_deg=0. if t==0 else 30.,camera_shape='cube',camera_valid=True))
        self.assertFalse(loop.goal_observed)


if __name__=='__main__':unittest.main()
