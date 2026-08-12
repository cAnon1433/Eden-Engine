#include "Camera.h"

#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>

namespace Eden
{
    Camera::Camera(glm::vec3 startPosition)
        : Position(startPosition)
        , WorldUp(0.0f, 1.0f, 0.0f)
    {
        UpdateVectors();
    }

    void Camera::ProcessKeyboard(CameraMovement direction, float deltaTime)
    {
        float velocity = MovementSpeed * deltaTime;

        switch (direction)
        {
            case CameraMovement::Forward:  Position += Front * velocity; break;
            case CameraMovement::Backward: Position -= Front * velocity; break;
            case CameraMovement::Left:     Position -= Right * velocity; break;
            case CameraMovement::Right:    Position += Right * velocity; break;
            case CameraMovement::Up:       Position += WorldUp * velocity; break;
            case CameraMovement::Down:     Position -= WorldUp * velocity; break;
        }
    }

    void Camera::ProcessMouseMovement(float xOffset, float yOffset)
    {
        Yaw += xOffset * MouseSensitivity;
        Pitch += yOffset * MouseSensitivity;

        // Clamp so the camera can't flip upside down at the poles.
        Pitch = std::clamp(Pitch, -89.0f, 89.0f);

        UpdateVectors();
    }

    void Camera::UpdateVectors()
    {
        glm::vec3 front;
        front.x = std::cos(glm::radians(Yaw)) * std::cos(glm::radians(Pitch));
        front.y = std::sin(glm::radians(Pitch));
        front.z = std::sin(glm::radians(Yaw)) * std::cos(glm::radians(Pitch));

        Front = glm::normalize(front);
        Right = glm::normalize(glm::cross(Front, WorldUp));
        Up = glm::normalize(glm::cross(Right, Front));
    }

    glm::mat4 Camera::GetViewMatrix() const
    {
        return glm::lookAt(Position, Position + Front, Up);
    }

    glm::mat4 Camera::GetProjectionMatrix(float aspectRatio) const
    {
        glm::mat4 proj = glm::perspective(glm::radians(FovDegrees), aspectRatio, NearPlane, FarPlane);
        // GLM was written for OpenGL's clip space (Y up); Vulkan's is Y
        // down. Flip it here rather than fighting it in every shader.
        proj[1][1] *= -1.0f;
        return proj;
    }
}
