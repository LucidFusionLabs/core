varying vec3 eye, dir;

uniform float fov_x, fov_y;  // Field of vision.

float fov2scale(float fov) { return tan(fov/2.0); }

// Draw an untransformed rectangle covering the whole screen.
// Get camera position and interpolated directions from the modelview matrix.
void main() {
  gl_Position = gl_Vertex;
  eye = vec3(gl_ModelViewMatrix[3]);
  dir = vec3(gl_ModelViewMatrix * vec4(
    fov2scale(fov_x)*gl_Vertex.x, fov2scale(fov_y)*gl_Vertex.y, 1, 0) );
}
