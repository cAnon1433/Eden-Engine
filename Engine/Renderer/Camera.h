#pragma once

#include <glm/glm.hpp>

namespace Eden
{
    enum class CameraMovement
    {
        Forward, Backward, Left, Right, Up, Down
    };

    // Free-fly camera: WASD + Space/Ctrl to move, mouse to look around.
    // Yaw/pitch driven (no roll) - standard FPS-style camera, not a full
    // quaternion camera. Good enough until gimbal lock near the poles
    // actually becomes a problem for what you're doing.
    class Camera
    {
    public:
        Camera(glm::vec3 startPosition = glm::vec3(0.0f, 0.0f, 3.0f));

        void ProcessKeyboard(CameraMovement direction, float deltaTime);
        void ProcessMouseMovement(float xOffset, float yOffset);

        glm::mat4 GetViewMatrix() const;
        glm::mat4 GetProjectionMatrix(float aspectRatio) const;

        float MovementSpeed = 2.5f;
        float MouseSensitivity = 0.1f;
        float FovDegrees = 60.0f;
        float NearPlane = 0.1f;
        float FarPlane = 100.0f;

    private:
        void UpdateVectors();

    public:
        glm::vec3 Position;
        glm::vec3 Front;
        glm::vec3 Up;
        glm::vec3 Right;
        glm::vec3 WorldUp;

        float Yaw = -90.0f;   // facing -Z initially
        float Pitch = 0.0f;
    };
}
