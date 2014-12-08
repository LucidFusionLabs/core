uniform mat4 ModelviewProjection;
uniform vec4 DefaultColor;
attribute vec4 Position;

uniform float time;
varying float TimeOut;

uniform vec2 resolution;
varying vec2 ResolutionOut;

void main() {
    TimeOut = time;
    ResolutionOut = resolution;
#ifdef LFL_GLES2
    gl_Position = ModelviewProjection * Position;
#else
    gl_Position = gl_Vertex;
#endif
}
