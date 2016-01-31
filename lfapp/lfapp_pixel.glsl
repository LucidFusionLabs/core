#ifndef GL_ES
#define lowp
#define highp
#endif

varying lowp vec4 DestinationColor;

#ifdef TEX2D
uniform bool TexCoordEnabled;
varying lowp vec2 TexCoordOut;
uniform lowp sampler2D iChannel0;
#endif

#ifdef TEXCUBE
uniform lowp samplerCube CubeTexture;
varying lowp vec3 CubeCoordOut;
#endif

void main(void) {
#ifdef TEXCUBE
  gl_FragColor = DestinationColor * textureCube(CubeTexture, CubeCoordOut);
#else // TEXCUBE

#ifdef TEX2D
  gl_FragColor = TexCoordEnabled ? DestinationColor * texture2D(iChannel0, TexCoordOut) : DestinationColor; 
#else // TEX2D
  gl_FragColor = DestinationColor; 
#endif // TEX2D
  // gl_FragColor = pow(gl_FragColor, vec4(1.0/2.2)); // gamma correction
#endif // TEXCUBE
}
