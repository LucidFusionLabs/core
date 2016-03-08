// Alien Tech by Kali

#define SHOWLIGHT //comment this line if you find the moving ligth annoying like Dave :D
#define BLINKINGLIGHTS 1.

// change this to tweak the fractal
#define c vec2(2.,4.5) 

//other cool params (comment default then uncomment one of this):
//#define c vec2(1.,5.)
//#define c vec2(4.,.5)
//#define c vec2(4.-length(p)*.2)
//#define c vec2(abs(sin(p.y*2.)),5.) //love this one with blinking


float ti=iGlobalTime;
vec3 ldir;
float ot;
float blur;


// 2D fractal based on Mandelbox
float formula(vec2 p) {
	vec2 t = vec2(sin(ti * .3) * .1 + ti * .05, ti * .1); // move fractal
	t+= iMouse.xy / iResolution.xy;
	p=abs(.5 - fract(p * .4 + t)) * 1.3; // tiling
	ot=1000.; 
	float l, expsmo;
	float aav=0.;
	l=0.; expsmo=0.;
	for (int i = 0; i < 11; i++) { 
		p = abs(p + c) - abs(p - c) - p; 
		p/= clamp(dot(p, p), .0007, 1.);
		p = p* -1.5 + c;
		if ( mod(float(i), 2.) < 1. ) { // exponential smoothing calc, with iteration skipping
			float pl = l;
			l = length(p);
			expsmo+= exp(-1. / abs(l - pl));
			ot=min(ot, l);
		}
	}
	return expsmo;
}

vec3 light(vec2 p, vec3 col) {
	
	// calculate normals based on horizontal and vertical vectors being z the formula result
	vec2 d = vec2(0., .003);
	float d1 = formula(p - d.xy) - formula(p+d.xy);
	float d2 = formula(p - d.yx) - formula(p+d.yx);	
  	vec3 n1 = vec3(0.    , d.y*2., -d1*.05);
  	vec3 n2 = vec3(d.y*2., 0.    , -d2*.05);
  	vec3 n = normalize(cross(n1, n2));

	// lighting
	float diff = pow( max(0., dot(ldir, n)) , 2.) + .2; // lambertian diffuse + ambient
	vec3 r = reflect(vec3(0.,0.,1.), ldir); // half vector
	float spec = pow( max(0., dot(r,n)) , 30.); // specular
  	return diff*col + spec*.8;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = fragCoord.xy / iResolution.xy - .5;
	vec2 aspect = vec2(iResolution.x / iResolution.y, 1.);
	uv*= aspect;
	vec2 pixsize = .25 / iResolution.xy * aspect; // pixel size for antialias
	float sph = length(uv); sph = sqrt(1. - sph*sph) * 1.5; // curve for spheric distortion
	uv = normalize(vec3(uv, sph)).xy * 1.3; // normalize back to 2D and scale (zoom level)
	pixsize = normalize(vec3(pixsize, sph)).xy * 1.3; // the same with pixsize for proper AA

	#ifdef SHOWLIGHT
	vec3 lightpos = vec3(sin(ti), cos(ti * .5), - .7); // moving light
	#else
	vec3 lightpos=vec3(0.,0.,-1.); // static light
	#endif

	lightpos.xy*= aspect * .25; // correct light coordinates
	vec3 col = vec3(0.);
	float lig = 0.;
	float titila = texture2D(iChannel0, vec2(ti * .25)).x; // for light intensity variation

	// AA loop
	for ( float aa = 0.; aa<9. ; aa++ ) { 
		vec2 aacoord = floor( vec2(aa/3., mod(aa,3.)) ); // get coord offset for AA sample
		vec2 p = uv + aacoord * pixsize; 
		ldir = normalize(vec3(p, .0) + lightpos); // get light direction
		float k = clamp(formula(p) * .25, .8, 1.4); // get value for colors in the desired range
		col+= light(p, vec3(k, k*k, k*k*k)); // accumulate surface color (a gradient trick)
		lig+= max(0., 2. - ot) / 2.; // accumulate orbit trap (yellow lights, shared "ot" var)
	}

	col*= .2; // correct brightness
	vec2 luv = uv + lightpos.xy; // uv shift by light coords

	// min amb light + spotlight with falloff * varying intensity
	col*= .07 + pow( max(0., 1. - length(luv) * .5), 9. ) * (1. - titila * .3);
	
	// rotating star light
	float star = abs(1.5708 - mod(atan(luv.x, luv.y) *3. - ti * 10., 3.1416)) * .02 - .05;
	#ifdef SHOWLIGHT
	col+= pow( max(0.,.3 - length(luv * 1.5) - star) / .3 , 5.) * (1. - titila * .5);
	#endif
	
	// yellow lights
	col+= pow(lig * .12, 15.) * vec3(1.,.9,.3) * (.8 + BLINKINGLIGHTS * sin(ti * 5. - uv.y * 10.) * .6);

	fragColor = BlendChannels(SampleChannel(iChannel0), vec4(col, 1.0));
}
