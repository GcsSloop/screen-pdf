"""Training-only guard: pixel normalization, old-correct masking and direction."""
import numpy as np
import torch

from training.train_screen_corner_single import positive_guard_loss, teacher_point_errors


def test_teacher_errors_use_pixel_bbox_diagonal_on_wide_images():
    target = np.array([[.1,.1],[.9,.1],[.9,.9],[.1,.9]], dtype=np.float32)
    prediction = target.copy()
    prediction[0,0] += .005
    size = np.array([4000,1000], dtype=np.float32)
    expected = 20 / np.hypot(3200,800)
    np.testing.assert_allclose(teacher_point_errors(prediction,target,size)[0],expected,rtol=1e-5)
    assert (teacher_point_errors(prediction,target,size) <= .01).all()


def test_guard_masks_teacher_errors_and_does_not_penalize_student_improvements():
    target=torch.tensor([[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]])
    sizes=torch.tensor([[4000.,1000.]])
    teacher=target.clone(); teacher[:,:,0] += .004
    student=target.clone(); student[0,0,0] += .008
    student[0,1,0] += .002  # student closer to GT: must not pull back
    student[0,2,0] += .1    # masked teacher failure: must not constrain
    student.requires_grad_()
    loss=positive_guard_loss(student,target,teacher,torch.tensor([[1.,1.,0.,0.]]),sizes)
    loss.backward()
    assert loss > 0 and student.grad[0,0,0] > 0
    assert torch.count_nonzero(student.grad[0,1:]) == 0
    zero=positive_guard_loss(student,target,teacher,torch.zeros(1,4),sizes)
    assert zero.item() == 0 and torch.isfinite(zero)
