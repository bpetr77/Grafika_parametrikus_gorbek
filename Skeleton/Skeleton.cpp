//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Bakó Péter
// Neptun : OTQVE5
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"


const char* const vertexSource = R"(
	#version 330
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

const char* const fragmentSource = R"(
	#version 330
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram;

class Camera2D {
	vec2 wCenter;
	vec2 wSize;
public:
	Camera2D() : wCenter(0, 0), wSize(30, 30) {}
	mat4 V() { return TranslateMatrix(-wCenter); }
	mat4 P() {
		return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y));
	}
	mat4 Vinv() {
		return TranslateMatrix(wCenter);
	}
	mat4 Pinv() {
		return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2));
	}
	void Zoom(float s) {
		wSize = wSize * s;
	}
	void Pan(vec2 t) {
		wCenter = wCenter + t;
	}
	void Pan12(vec2 t) { wCenter = wCenter + t / wSize.x; }
};

Camera2D camera;
const int nTessVertices = 100;
class Curve {
	unsigned int vao = 0, vbo = 0, nVtx = 0;
protected:
	std::vector<vec2> cps;
	std::vector<float> ts;
public:
	void create() {
		glGenVertexArrays(1, &vao); glBindVertexArray(vao);
		glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}
	void UpdateGPU() {
		glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo);
		std::vector<vec2> vtx = GenVertexData(); nVtx = vtx.size();
		glBufferData(GL_ARRAY_BUFFER, nVtx * sizeof(vec2), &vtx[0], GL_STATIC_DRAW);
	}

	std::vector<vec2> GenVertexData() {
		std::vector<vec2> vertices;
		for (int i = 0; i <= nTessVertices; ++i) {
			float t = (float)i / nTessVertices;
			vec2 point = r(t);
			if (point.x != 999 || point.y != 999) {
				vertices.push_back(point);
			}
		}
		return vertices;
	}

	int PickControlPoint(float cX, float cY) {
		vec4 hVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		vec2 mousePos = vec2(hVertex.x, hVertex.y);
		for (unsigned int i = 0; i < cps.size(); i++) {
			if (length(mousePos - cps[i]) < 0.1) {
				return i;
			}
		}
		return -1;
	}
	void MoveControlPoint(int index, float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		cps[index] = vec2(wVertex.x, wVertex.y);
	}

	virtual void AddControlPoint(vec2 cp) = 0;

	virtual vec2 r(float t) = 0;

	void Draw(int type, vec3 color) {
		if (cps.size() == 0) return;

		mat4 MVPtransf = camera.P() * camera.V();
		gpuProgram.setUniform(MVPtransf, "MVP");

		UpdateGPU(); glLineWidth(2.0f);
		gpuProgram.setUniform(color, "color");
		glBindVertexArray(vao); glDrawArrays(type, 0, nVtx);

		vec3 controlPointColor = vec3(1.0f, 0.0f, 0.0f);
		gpuProgram.setUniform(controlPointColor, "color");
		glPointSize(10);
		glBufferData(GL_ARRAY_BUFFER, cps.size() * sizeof(vec2), &cps[0], GL_STATIC_DRAW);
		glDrawArrays(GL_POINTS, 0, cps.size());
	}
	void Draw1(int type, vec3 color) {
		if (cps.size() == 0) return;

		//mat4 MVPtransf = camera.P() * camera.V();
		mat4 MVPtransf = camera.V() * camera.P();
		gpuProgram.setUniform(MVPtransf, "MVP");

		vec3 controlPointColor = vec3(1.0f, 0.0f, 0.0f);
		gpuProgram.setUniform(controlPointColor, "color");
		glPointSize(10);
		glBufferData(GL_ARRAY_BUFFER, cps.size() * sizeof(vec2), &cps[0], GL_STATIC_DRAW);
		glDrawArrays(GL_POINTS, 0, cps.size());

		if (cps.size() >= 2) {
			UpdateGPU(); glLineWidth(2.0f);
			gpuProgram.setUniform(color, "color");
			glBindVertexArray(vao); glDrawArrays(type, 0, nVtx);
		}
	}
};


class BezierCurve : public Curve {
	float B(size_t i, float t) {
		float choose = 1;
		for (size_t j = 1; j <= i; j++) choose *= (float)(cps.size() - j) / j;
		return choose * powf(t, i) * powf(1 - t, cps.size() - 1 - i);
	}
public:

	void AddControlPoint(vec2 cp) {
		//vec4 wVertex = vec4(cp.x, cp.y, 0, 1) * camera.Pinv() * camera.Vinv();
		vec4 wVertex = vec4(cp.x, cp.y, 0, 1) * camera.Vinv() * camera.Pinv();
		cps.push_back(vec2(wVertex.x, wVertex.y));
	}

	BezierCurve() {
		create();
	}

	vec2 r(float t) {
		vec2 rt(0, 0);
		for (size_t i = 0; i < cps.size(); i++) rt = rt + cps[i] * B(i, t);
		return rt;
	}
};


class LagrangeCurve : public Curve {
	float L(int i, float t) {
		float Li = 1.0f;
		for (size_t j = 0; j < cps.size(); j++)
			if (j != i) Li *= (t - ts[j]) / (ts[i] - ts[j]);
		return Li;
	}
public:
	LagrangeCurve() {
		create();
	}

	vec2 r(float t) {
		if (ts.empty()) {
			return vec2(0, 0);
		}
		t *= ts.back();
		vec2 rt(0, 0);
		for (size_t i = 0; i < cps.size(); i++) rt = rt + cps[i] * L(i, t);
		return rt;
	}
	void AddControlPoint(vec2 cp) {
		//vec4 wVertex = vec4(cp.x, cp.y, 0, 1) * camera.Pinv() * camera.Vinv();
		vec4 wVertex = vec4(cp.x, cp.y, 0, 1) * camera.Vinv() * camera.Pinv();
		vec2 transformedCp = vec2(wVertex.x, wVertex.y);


		cps.push_back(transformedCp);

		float totalLength = 0.0f;
		for (size_t i = 1; i < cps.size(); ++i) {
			totalLength += length(cps[i] - cps[i - 1]);
		}

		ts.clear();
		float accumulatedLength = 0.0f;
		for (size_t i = 0; i < cps.size(); ++i) {
			if (i > 0) {
				accumulatedLength += length(cps[i] - cps[i - 1]);
			}
			ts.push_back(accumulatedLength / totalLength);
		}

		UpdateGPU();
	}
};


class CatmullRomCurve : public Curve {
	vec2 Hermite(vec2 p0, vec2 v0, float t0, vec2 p1, vec2 v1, float t1, float t) {
		float t01 = t - t0;
		vec2 a0 = p0;
		vec2 a1 = v0;
		vec2 a2 = (3 * (p1 - p0) / powf(t1 - t0, 2)) - ((v1 + 2 * v0) / (t1 - t0));
		vec2 a3 = (2 * (p0 - p1) / powf(t1 - t0, 3)) + ((v1 + v0) / powf(t1 - t0, 2));
		return a3 * powf(t01, 3) + a2 * powf(t01, 2) + a1 * t01 + a0;
	}
	float tension = 0.0f;
public:
	CatmullRomCurve() {
		create();
	}
	void IncreaseTension() {
		tension += 0.1f;
	}
	void DecreaseTension() {
		tension -= 0.1f;
	}

	void AddControlPoint(vec2 cp) {
		//vec4 wVertex = vec4(cp.x, cp.y, 0, 1) * camera.Pinv() * camera.Vinv();
		vec4 wVertex = vec4(cp.x, cp.y, 0, 1) * camera.Vinv() * camera.Pinv();
		vec2 transformedCp = vec2(wVertex.x, wVertex.y);

		cps.push_back(transformedCp);
		if (cps.size() == 1) {
			ts.push_back(0.0f);
		}
		else {
			ts.push_back(0.0f);
			float totalLength = 0.0f;
			for (size_t i = 1; i < cps.size(); ++i) {
				totalLength += length(cps[i] - cps[i - 1]);
			}

			ts.clear();
			float accumulatedLength = 0.0f;
			for (size_t i = 0; i < cps.size(); ++i) {
				if (i > 0) {
					accumulatedLength += length(cps[i] - cps[i - 1]);
				}
				ts.push_back(accumulatedLength / totalLength);
			}
		}
		UpdateGPU();
	}
	vec2 r1(float t) {
		if (ts.empty()) {
			return vec2(999, 999);
		}
		for (size_t i = 0; i < ts.size() - 1; i++) {
			if (ts[i] <= t && t <= ts[i + 1]) {
				vec2 v0 = (i > 0) ? (1 - tension) * 0.5f * ((cps[i + 1] - cps[i]) / (ts[i + 1] - ts[i]) + (cps[i] - cps[i - 1]) / (ts[i] - ts[i - 1])) : (1 - tension) * 0.5f * ((cps[i + 1] - cps[i]) / (ts[i + 1] - ts[i]));
				vec2 v1 = (i < cps.size() - 2) ? (1 - tension) * 0.5f * ((cps[i + 2] - cps[i + 1]) / (ts[i + 2] - ts[i + 1]) + (cps[i + 1] - cps[i]) / (ts[i + 1] - ts[i])) : (1 - tension) * 0.5f * ((cps[i + 1] - cps[i]) / (ts[i + 1] - ts[i]));
				return Hermite(cps[i], v0, ts[i], cps[i + 1], v1, ts[i + 1], t);
			}
		}
		return cps[0];
	}
	vec2 r(float t) {
		if (ts.empty()) {
			return vec2(999, 999);
		}
		for (size_t i = 0; i < ts.size() - 1; i++) {
			if (ts[i] <= t && t <= ts[i + 1]) {
				vec2 v0 = (i > 0) ? (1 - tension) * 0.5f * (((cps[i + 1] - cps[i]) / (ts[i + 1] - ts[i])) + ((cps[i] - cps[i - 1]) / (ts[i] - ts[i - 1]))) : (1 - tension) * 0.5f * ((cps[i + 1] - cps[i]) / (ts[i + 1] - ts[i]));
				vec2 v1 = (i < cps.size() - 2) ? (1 - tension) * 0.5f * (((cps[i + 2] - cps[i + 1]) / (ts[i + 2] - ts[i + 1])) + ((cps[i + 1] - cps[i]) / (ts[i + 1] - ts[i]))) : (1 - tension) * 0.5f * ((cps[i + 1] - cps[i]) / (ts[i + 1] - ts[i]));
				return Hermite(cps[i], v0, ts[i], cps[i + 1], v1, ts[i + 1], t);
			}
		}
		return cps[0];
	}
};


enum CurveType {
	Lagrange,
	Bezier,
	CatmullRom
};
CurveType curveType;

Curve* curve = nullptr;
int selectedPoint = -1;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

void onDisplay() {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);

	if (curve != nullptr) {
		curve->Draw(GL_LINE_STRIP, vec3(1, 1, 0));
		curve->UpdateGPU();
	}

	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	float zoomFactor = 1.1f;
	float panDistance = 0.0666666666f;
	switch (key) {
	case 'l':
		delete curve;
		curve = new LagrangeCurve();
		curveType = Lagrange;
		break;
	case 'b':
		delete curve;
		curve = new BezierCurve();
		curveType = Bezier;
		break;
	case 'c':
		delete curve;
		curve = new CatmullRomCurve();
		curveType = CatmullRom;
		break;
	case 'Z':
		camera.Zoom(zoomFactor);
		break;
	case 'z':
		camera.Zoom(1 / zoomFactor);
		break;
	case 'P':
		camera.Pan(vec2(panDistance, 0));
		break;
	case 'p':
		camera.Pan(vec2(-panDistance, 0));
		break;
	case 'T':
		if (curveType == CatmullRom) {
			((CatmullRomCurve*)curve)->IncreaseTension();
		}
		break;
	case 't':
		if (curveType == CatmullRom) {
			((CatmullRomCurve*)curve)->DecreaseTension();
		}
		break;
	default:
		return;
	}
	glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	if (selectedPoint != -1) {
		curve->MoveControlPoint(selectedPoint, cX, cY);
		glutPostRedisplay();
	}
}

void onMouse(int button, int state, int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && curveType == Lagrange) {
		if (curve != nullptr) {
			curve->AddControlPoint(vec2(cX, cY));
			curve->UpdateGPU();
		}
		glutPostRedisplay();
	}
	else if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && curveType == Bezier) {
		if (curve != nullptr) {
			curve->AddControlPoint(vec2(cX, cY));
			curve->UpdateGPU();
		}
		glutPostRedisplay();
	}
	else if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && curveType == CatmullRom) {
		if (curve != nullptr) {
			curve->AddControlPoint(vec2(cX, cY));
			curve->UpdateGPU();
		}
		glutPostRedisplay();
	}
	else if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
		selectedPoint = curve->PickControlPoint(cX, cY);
	}
	else if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP) {
		selectedPoint = -1;
	}
}

void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME);
}