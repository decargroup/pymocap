from pymocap import MocapTrajectory
import numpy as np
from typing import List 
import navlie as nav
np.set_printoptions(precision=3, suppress=True, linewidth=200)

filename = "data/ifo003_bag_2022-12-08-08-43-33.bag"
agent = "ifo003"
mocap = MocapTrajectory.from_bag(filename, agent)

# Test the MocapTrajectory class's plot method.
def test_plot():
    mocap.plot()

# Test that raw quaternion data is unit norm.
# I guess this is more of a test of ROS data
def test_quat_norm():
    assert np.allclose(np.linalg.norm(mocap.raw_quaternion, axis=1), 1)

# Test interpolating quaternion data at exactly the same time as the mocap data.
def test_interpolate_quat():
    t = mocap.stamps[10]
    quat = mocap.quaternion(t)
    # There will be some smoothing error since interpolation is not exact
    assert np.allclose(quat, mocap.raw_quaternion[10], atol=1e-2)

def test_mocap_to_navlie_se3():
    data: List[nav.lib.SE3State]  = mocap.to_navlie(mocap.stamps, "SE3")
    assert len(data) == len(mocap.stamps)
    assert data[0].stamp == mocap.stamps[0]
    assert data[0].state_id == agent
    assert np.allclose(data[0].value, mocap.pose_matrix(mocap.stamps[0]))

def test_mocap_to_navlie_se23():
    data: List[nav.lib.SE23State]  = mocap.to_navlie(mocap.stamps, "SE23")
    assert len(data) == len(mocap.stamps)
    assert data[0].stamp == mocap.stamps[0]
    assert data[0].state_id == agent
    assert np.allclose(data[0].value, mocap.extended_pose_matrix(mocap.stamps[0]))

def test_mocap_to_navlie_imu():
    data: List[nav.lib.IMUState]  = mocap.to_navlie(mocap.stamps, "IMU")
    assert len(data) == len(mocap.stamps)
    assert data[0].stamp == mocap.stamps[0]
    assert data[0].state_id == agent
    assert np.allclose(data[0].pose, mocap.extended_pose_matrix(mocap.stamps[0]))
    assert np.allclose(data[0].bias_accel, np.array([0,0,0]))
    assert np.allclose(data[0].bias_gyro, np.array([0,0,0]))
    assert np.allclose(data[0].velocity, mocap.velocity(mocap.stamps[0]))
    assert np.allclose(data[0].position, mocap.position(mocap.stamps[0]))
    assert np.allclose(data[0].attitude, mocap.rot_matrix(mocap.stamps[0]))

if __name__ == "__main__":
    test_mocap_to_navlie_imu()