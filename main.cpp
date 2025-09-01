// ============================================================================
// Hydra 实时同步：MuJoCo ↔ OpenUSD （C++ 最小原型，带 GLFW OpenGL 上下文）
// ============================================================================
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformCommonAPI.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/base/gf/camera.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/glf/contextCaps.h>
#include <pxr/usdImaging/usdImagingGL/engine.h>
#include <mujoco/mujoco.h>
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <string>
#include "tinyxml2.h"

using namespace pxr;
using namespace tinyxml2;

bool ExportUsdStageToMjcf(UsdStageRefPtr stage, const std::string& outputDir, const std::string& xmlFile)
{
    // 2. 打开 XML 文件
    std::ofstream xml(xmlFile);
    if (!xml) {
        std::cerr << "Failed to open XML file: " << xmlFile << std::endl;
        return false;
    }

    xml << "<mujoco model=\"usd_scene\">\n  <asset>\n";

    int meshCount = 0;

    // 3. 遍历 Stage 所有 prim
    for (UsdPrim prim : stage->Traverse()) {
        UsdGeomMesh mesh(prim);
        if (!mesh)
            continue;
        VtVec3fArray extent;
        mesh.UsdGeomBoundable::ComputeExtent(UsdTimeCode::Default(), &extent);
        
        GfVec3f min = extent[0];
        GfVec3f max = extent[1];
        GfVec3f size = max - min;
        if (size[2] < 1e-8)
            continue;

        const char* Name = prim.GetName().GetText();
        const char* Path = prim.GetPath().GetText();

        std::string meshName = "mesh_" + std::to_string(meshCount);
        std::string objFile = outputDir + meshName + ".obj";

        // 获取顶点
        VtArray<GfVec3f> points;
        mesh.GetPointsAttr().Get(&points);

        // 获取面
        VtArray<int> faceCounts;
        mesh.GetFaceVertexCountsAttr().Get(&faceCounts);
        VtArray<int> faceIndices;
        mesh.GetFaceVertexIndicesAttr().Get(&faceIndices);

        // 写 OBJ 文件
        std::ofstream obj(objFile);
        if (!obj) {
            std::cerr << "Failed to write obj: " << objFile << std::endl;
            continue;
        }

        for (const auto& p : points) {
            obj << "v " << p[0] << " " << p[1] << " " << p[2] << "\n";
        }

        size_t idx = 0;
        for (size_t f = 0; f < faceCounts.size(); ++f) {
            int c = faceCounts[f];
            if (c == 3) {
                obj << "f " << faceIndices[idx]+1 << " " << faceIndices[idx+1]+1 << " " << faceIndices[idx+2]+1 << "\n";
            } else if (c == 4) {
                obj << "f " << faceIndices[idx]+1 << " " << faceIndices[idx+1]+1 << " " << faceIndices[idx+2]+1 << "\n";
                obj << "f " << faceIndices[idx]+1 << " " << faceIndices[idx+2]+1 << " " << faceIndices[idx+3]+1 << "\n";
            } else {
                std::cerr << "Skipping polygon with " << c << " vertices\n";
            }
            idx += c;
        }
        obj.close();

        // 写 XML <mesh>
        xml << "    <mesh name=\"" << meshName << "\" file=\"" << objFile << "\"/>\n";

        ++meshCount;
    }

    xml << "  </asset>\n  <worldbody>\n";

    // 为每个 mesh 生成一个 <body><geom>
    for (int i = 0; i < meshCount; ++i) {
        std::string meshName = "mesh_" + std::to_string(i);
        xml << "    <body name=\"" << meshName << "_body\" pos=\"0 0 0\">\n"
            << "      <geom type=\"mesh\" mesh=\"" << meshName << "\" />\n"
            << "    </body>\n";
    }

    xml << "  </worldbody>\n</mujoco>\n";
    xml.close();

    std::cout << "✅ Exported " << meshCount << " meshes to MJCF XML: " << xmlFile << std::endl;
    return true;
}

// 简单桥接：MuJoCo → USD
class MjUsdBridge {
public:
    MjUsdBridge(const std::string& usd_path)
    {
        stage = UsdStage::Open(usd_path);

        XMLDocument doc;
        XMLElement* mujoco = doc.NewElement("mujoco");
        doc.InsertFirstChild(mujoco);
        XMLElement* worldbody = doc.NewElement("worldbody");
        mujoco->InsertEndChild(worldbody);
        worldbody->InsertEndChild(Export(stage->GetPrimAtPath(SdfPath("/")), doc));
        XMLError eResult = doc.SaveFile("scene.xml");
        if (eResult != XML_SUCCESS)
            std::cerr << "Error saving XML\n";

        // 2. 初始化 MuJoCo 模型
        model = mj_loadXML("scene.xml", nullptr, nullptr, 0);
        data = mj_makeData(model);

        //bodyNames = {"mesh_0_body", "mesh_1_body"};
        //primPaths = {SdfPath("/World/mesh_0"), SdfPath("/World/mesh_1")};
    }

    ~MjUsdBridge()
    {
        if (data)
            mj_deleteData(data);
        if (model)
            mj_deleteModel(model);
    }

    XMLElement* Export(const UsdPrim prim, XMLDocument& doc)
    {
        XMLElement* elem = doc.NewElement("body");
        for (UsdPrim child : prim.GetChildren())
        {
            XMLElement* child_elem = Export(child, doc);
            elem->InsertEndChild(child_elem);
        }
        return elem;
    }

    UsdStageRefPtr GetStage() { return stage; }

    void StepAndSync(double time, int frame)
    {
        mj_step(model, data);
        
        if(bodyNames.size() != primPaths.size())
        {
            std::cerr << "bodyNames and primPaths size mismatch!" << std::endl;
            return;
        }
        for(size_t i=0; i<bodyNames.size(); ++i)
        {
            int id = mj_name2id(model, mjOBJ_BODY, bodyNames[i].c_str());
            if(id < 0)
                continue;
            const double* pos = data->xpos + 3*id;
            const double* quat = data->xquat + 4*id; // w x y z
            GfMatrix4d mat;
            GfQuatd q(quat[0], quat[1], quat[2], quat[3]);
            mat.SetRotate(q);
            mat.SetTranslateOnly(GfVec3d(pos[0], pos[1], pos[2]));
            UsdGeomXform x = UsdGeomXform::Get(stage, primPaths[i]);
            if (!x.GetTransformOp())
                x.AddTransformOp();
            x.GetTransformOp().Set(mat, time);
        }
    }

private:
    mjModel* model = nullptr;
    mjData* data = nullptr;
    std::vector<std::string> bodyNames;
    std::vector<SdfPath> primPaths;
    UsdStageRefPtr stage;
    UsdGeomXform rootX;
};
#define WIDTH 1024
#define HEIGHT 768

double lookAtDistance = 6.0;
double yaw = 7.0;
double pitch = 0.0;
double rollX = 0.0;
double rollY = 0.0;
double offsetX = 0.0;
double offsetY = 0.0;
float domeExposure = 1.0f;

int currentDelegate = 0;
int newDelegate = 0;

std::string currentFilename = "";
std::string newFilename = "";

bool animate = true;

int totalCubes = 0;

bool fullscreen = false;

int recentButton = GLFW_GAMEPAD_BUTTON_LAST;

bool showHelp = true;
bool highlight = true;

float eyesHeight = 0.0f;
float focalLength = 30.f;

float positionMultiplier = 5.0f;
float rotationMultiplier = 0.2f;
float heightMultiplier = 2.0f;
float distanceMultiplier = 3.0f;
int main(int argc, char** argv)
{
	if (!glfwInit())
	{
		std::cout << "Failed initializing glfw" << std::endl;
		return 1;
	}
    //glfwSetErrorCallback(error_callback);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "simplecube", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed creating a glfw window with OpenGL, retrying without it" << std::endl;
        return 1;
    }
    //glfwSetKeyCallback(window, key_callback);
    //glfwSetDropCallback(window, drop_callback);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    pxr::GlfContextCaps::InitInstance();

    std::unique_ptr<class pxr::UsdImagingGLEngine> engine;
    pxr::GfCamera camera;
    pxr::GfMatrix4d cameraTransform;
    pxr::GfVec3d cameraPivot(0,eyesHeight,0);

    pxr::GfMatrix4d viewMatrix;
    pxr::GfMatrix4d projectionMatrix;
    pxr::GfFrustum frustum;

    pxr::UsdImagingGLRenderParams renderParams;

    MjUsdBridge bridge(argv[1]);
    UsdStageRefPtr stage = bridge.GetStage();

    pxr::SdfPathVector excludedPaths;
    engine.reset(new pxr::UsdImagingGLEngine(
        stage->GetPseudoRoot().GetPath(), excludedPaths));

    std::cout << "Available Hydra Delegates:" << std::endl;
#if PXR_VERSION >= 2311
    const auto& renderDelegates = engine->GetRendererPlugins();
#else
    auto renderDelegates = engine->GetRendererPlugins();
#endif
    for (int i = 0; i < renderDelegates.size(); i++)
    {
        std::cout << renderDelegates[i].GetString() << std::endl;
    }
    bool enabled = engine->SetRendererPlugin(renderDelegates[1]);

    int frame = 0;

    pxr::GfVec4f clearColor(0.18f, 0.18f, 0.18f, 1.0f);

    pxr::GfVec2f joystickZeroLeft;
    pxr::GfVec2f joystickZeroRight;
    float joystickTriggerLeft = 0.0f;
    float joystickTriggerRight = 0.0f;
    if (glfwJoystickIsGamepad(GLFW_JOYSTICK_1))
    {
        GLFWgamepadstate state;
        if (glfwGetGamepadState(GLFW_JOYSTICK_1, &state))
        {
            std::cout << "Joystick/Gamepad found: " << glfwGetGamepadName(GLFW_JOYSTICK_1) << std::endl;
            joystickZeroLeft = pxr::GfVec2f(state.axes[GLFW_GAMEPAD_AXIS_LEFT_X], state.axes[GLFW_GAMEPAD_AXIS_LEFT_Y]);
            joystickZeroRight = pxr::GfVec2f(state.axes[GLFW_GAMEPAD_AXIS_RIGHT_X], state.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y]);
            joystickTriggerLeft = state.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER];
            joystickTriggerRight = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER];
        }
    }

    pxr::SdfPath selectedPrimPath;

    while (!glfwWindowShouldClose(window))
    {
        if(animate)
            frame++;
        // MuJoCo 推进一步并同步到 USD
        bridge.StepAndSync(frame * 0.01, frame);

        glfwMakeContextCurrent(window);

        glfwPollEvents();

        bool delegateSelectionMode = false;
        bool primLocked = false;

        if (glfwJoystickIsGamepad(GLFW_JOYSTICK_1))
        {
            GLFWgamepadstate state;
            if (glfwGetGamepadState(GLFW_JOYSTICK_1, &state))
            {
                pxr::GfVec2f stickLeft = pxr::GfVec2f(state.axes[GLFW_GAMEPAD_AXIS_LEFT_X], state.axes[GLFW_GAMEPAD_AXIS_LEFT_Y]);
                pxr::GfVec2f stickRight = pxr::GfVec2f(state.axes[GLFW_GAMEPAD_AXIS_RIGHT_X], state.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y]);

// marco to check if a gamepad button is pressed
#define GAMEPAD_BUTTON_PRESSED(btn) state.buttons[btn] == GLFW_PRESS
// macro to check if a gamepad button has been pressed once
#define GAMEPAD_BUTTON_ONCE(btn) state.buttons[btn] == GLFW_PRESS && recentButton != btn

                if (GAMEPAD_BUTTON_PRESSED(GLFW_GAMEPAD_BUTTON_X))
                {
                    primLocked = true;
                }
                if (GAMEPAD_BUTTON_PRESSED(GLFW_GAMEPAD_BUTTON_Y))
                {
                    delegateSelectionMode = true;
                }
                if (GAMEPAD_BUTTON_PRESSED(GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER))
                {
                    eyesHeight += heightMultiplier;
                    cameraPivot = pxr::GfVec3d(cameraPivot[0], eyesHeight, cameraPivot[2]);
                }
                if (GAMEPAD_BUTTON_PRESSED(GLFW_GAMEPAD_BUTTON_LEFT_BUMPER))
                {
                    eyesHeight -= heightMultiplier;
                    cameraPivot = pxr::GfVec3d(cameraPivot[0], eyesHeight, cameraPivot[2]);
                }
                if (GAMEPAD_BUTTON_ONCE(GLFW_GAMEPAD_BUTTON_DPAD_UP))
                {
                    if(delegateSelectionMode)
                        newDelegate = std::max(0, newDelegate - 1);
                }
                if (GAMEPAD_BUTTON_ONCE(GLFW_GAMEPAD_BUTTON_DPAD_DOWN))
                {
                    if(delegateSelectionMode)
                        newDelegate++;
                }

                if (GAMEPAD_BUTTON_PRESSED(GLFW_GAMEPAD_BUTTON_DPAD_LEFT))
                {
                    focalLength -= 0.2f;
                }
                if (GAMEPAD_BUTTON_PRESSED(GLFW_GAMEPAD_BUTTON_DPAD_RIGHT))
                {
                    focalLength += 0.2f;
                }

                // set recent button
                for (int bi = 0; bi < 15; ++bi)
                {
                    recentButton = GLFW_GAMEPAD_BUTTON_LAST;
                    if (bi != GLFW_GAMEPAD_BUTTON_Y && state.buttons[bi] == GLFW_PRESS )
                    {
                        recentButton = bi;
                        break;
                    }
                }
#undef GAMEPAD_BUTTON_PRESSED
#undef GAMEPAD_BUTTON_ONCE

                if ((joystickZeroLeft - stickLeft).GetLength() > 0.08)
                {
                    pxr::GfVec3d camDir = cameraTransform.ExtractTranslation() - cameraPivot;
                    camDir[1] = 0.0;
                    camDir.Normalize();
                    cameraPivot = pxr::GfVec3d(cameraPivot[0] + stickLeft[1] * camDir[0] * positionMultiplier, eyesHeight, cameraPivot[2] + stickLeft[1] * camDir[2] * positionMultiplier);
                    cameraPivot = pxr::GfVec3d(cameraPivot[0] + stickLeft[0] * camDir[2] * positionMultiplier, eyesHeight, cameraPivot[2] + stickLeft[0] * (-camDir[0]) * positionMultiplier);

                    if (primLocked && !selectedPrimPath.IsEmpty())
                    {
                        const pxr::UsdPrim& prim = stage->GetPrimAtPath(selectedPrimPath);
                        pxr::GfVec3d translate;
                        pxr::GfVec3f rotate;
                        pxr::GfVec3f scale;
                        pxr::GfVec3f pivot;
                        pxr::UsdGeomXformCommonAPI::RotationOrder rotOrder;
                        pxr::UsdGeomXformCommonAPI(prim).GetXformVectors(&translate, &rotate, &scale, &pivot, &rotOrder, pxr::UsdTimeCode::Default());

                        translate[0] += stickLeft[1] * camDir[0] * positionMultiplier + stickLeft[0] * camDir[2] * positionMultiplier;
                        translate[1] += 0.0;
                        translate[2] += stickLeft[1] * camDir[2] * positionMultiplier + stickLeft[0] * (-camDir[0]) * positionMultiplier;

                        // pxr::UsdGeomImageable(prim).ComputeLocalToWorldTransform(0);
                        const auto& sourcePrim = pxr::UsdGeomXform(prim);
                        sourcePrim.ClearXformOpOrder();
                        const auto& transformOp = sourcePrim.AddTransformOp();
                        auto m = pxr::GfMatrix4d().SetIdentity();
                        m *= pxr::GfMatrix4d().SetScale(scale);
                        m *= pxr::GfMatrix4d().SetRotate(pxr::GfRotation(pxr::GfVec3d(1, 0, 0), rotate[0]));
                        m *= pxr::GfMatrix4d().SetRotate(pxr::GfRotation(pxr::GfVec3d(0, 1, 0), rotate[1]));
                        m *= pxr::GfMatrix4d().SetRotate(pxr::GfRotation(pxr::GfVec3d(0, 0, 1), rotate[2]));
                        m *= pxr::GfMatrix4d().SetTranslate(translate);
                        transformOp.Set(m);

                    }
                }
                if ((joystickZeroRight - stickRight).GetLength() > 0.08)
                {
                    pitch += stickRight[0] * rotationMultiplier;
                    yaw += stickRight[1] * rotationMultiplier;
                }

                if (abs(joystickTriggerLeft - state.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER]) > 0.05)
                    lookAtDistance -= (state.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER] + 1.0) * distanceMultiplier;
                if (abs(joystickTriggerRight - state.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER]) > 0.05)
                    lookAtDistance += (state.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER] + 1.0) * distanceMultiplier;
                lookAtDistance = std::max(0.1, lookAtDistance);


            }
        }

        // set cube rotation
        //
        //cubeMeshOp.Set(float(frame));

        // get display size (inner display buffer)
        //
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        // get full-window size (borders included)
        //
        int window_w, window_h;
        glfwGetWindowSize(window, &window_w, &window_h);

        cameraTransform.SetIdentity();
        cameraTransform *= pxr::GfMatrix4d().SetTranslate(pxr::GfVec3d(-offsetX, -offsetY, 0.0));
        cameraTransform *= pxr::GfMatrix4d().SetTranslate(pxr::GfVec3d(0, 0, lookAtDistance));
        cameraTransform *= pxr::GfMatrix4d().SetRotate(pxr::GfRotation(pxr::GfVec3d(0, 0, 1), -rollX * 5.0));
        cameraTransform *= pxr::GfMatrix4d().SetRotate(pxr::GfRotation(pxr::GfVec3d(1, 0, 0), -yaw * 5.0));
        cameraTransform *= pxr::GfMatrix4d().SetRotate(pxr::GfRotation(pxr::GfVec3d(0, 1, 0), -pitch * 5.0));
        cameraTransform *= pxr::GfMatrix4d().SetRotate(pxr::GfRotation(pxr::GfVec3d(0, 0, 1), -rollY * 5.0));
        cameraTransform *= pxr::GfMatrix4d().SetTranslate(cameraPivot);

        camera.SetTransform(cameraTransform);
        frustum = camera.GetFrustum();
        double fovy = focalLength;
        double znear = 0.5;
        double zfar = 100000.0;
        const double aspectRatio = double(display_w) / double(display_h);
        frustum.SetPerspective(fovy, aspectRatio, znear, zfar);
        projectionMatrix = frustum.ComputeProjectionMatrix();
        viewMatrix = frustum.ComputeViewMatrix();

        //if (!pxr::UsdImagingGLEngine::IsColorCorrectionCapable())
        //    glEnable(GL_FRAMEBUFFER_SRGB_EXT);

        glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // setup viewport for 3d scene render
        glPushMatrix();
        {
            glViewport(0, 0, display_w, display_h);
            pxr::GfVec4d viewport(0, 0, display_w, display_h);

            // scene has lighting
            glEnable(GL_LIGHTING);
            glEnable(GL_LIGHT0);

            // Set engine properties and parameters
            engine->SetRenderBufferSize(pxr::GfVec2i(window_w, window_h));
            engine->SetSelectionColor(pxr::GfVec4f(1.0, 1.0, 0.0, 1.0));
            engine->SetCameraState(viewMatrix, projectionMatrix);
            engine->SetRenderViewport(viewport);
            engine->SetEnablePresentation(true);

            // update render params
            renderParams.frame = frame;
            renderParams.enableLighting = true;
            renderParams.enableSceneLights = true;
            renderParams.enableSceneMaterials = true;
            renderParams.showProxy = true;
            renderParams.showRender = false;
            renderParams.showGuides = false;
            renderParams.forceRefresh = false;
            renderParams.highlight = highlight;
            renderParams.enableUsdDrawModes = true;
            renderParams.drawMode = pxr::UsdImagingGLDrawMode::DRAW_SHADED_SMOOTH;
            renderParams.gammaCorrectColors = true;
            renderParams.clearColor = clearColor;
            //renderParams.enableIdRender = false;
            renderParams.enableSampleAlphaToCoverage = false;
            renderParams.complexity = 1.0f;

            // render all paths from root
            engine->Render(stage->GetPseudoRoot(), renderParams);
        }
        glPopMatrix();

        pxr::GfVec2d halfSize(1.0 / display_w, 1.0 / display_h);
        pxr::GfVec2d screenPoint(2.0 * ((display_w/2.0) / display_w) - 1.0, 2.0 * (1.0 - (display_h/2.0) / display_h) - 1.0);

        // Compute pick frustum.
        auto pickFrustum = frustum.ComputeNarrowedFrustum(screenPoint, halfSize);
        auto pickView = pickFrustum.ComputeViewMatrix();
        auto pickProj = pickFrustum.ComputeProjectionMatrix();

        pxr::GfVec3d selectionHitPoint;
        pxr::GfVec3d selectionHitNormal;

        if (!primLocked)
        {
            selectedPrimPath = pxr::SdfPath();
            if (highlight && engine->TestIntersection(
                pickView,
                pickProj,
                stage->GetPseudoRoot(),
                renderParams,
                &selectionHitPoint,
                &selectionHitNormal,
                &selectedPrimPath))
            {
                engine->SetSelected({ selectedPrimPath });
            }
            else
            {
                engine->ClearSelected();
            }
        }

        glClear(GL_DEPTH_BUFFER_BIT);

        // HUD
        glDisable(GL_LIGHTING);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glEnable(GL_LINE_SMOOTH);
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glPushMatrix();
        {
            glViewport(0, 0, display_w, display_h);
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
            
            // draw white cross
            glLineWidth(2.0f);
            glBegin(GL_LINES);
            glColor3f(1.0f, 1.0f, 1.0f);
            glVertex3f(-0.1, 0.0, 0.0);
            glVertex3f(0.1, 0.0, 0.0);
            glVertex3f(0.0, -0.1, 0.0);
            glVertex3f(0.0, 0.1, 0.0);
            glEnd();
            
            // draw text
            //pfResetTop();
            //pfPixelSize(2.0f);
            //pfDisplaySize(display_w, display_h);

            // show help
            // if (showHelp)
            // {
            //     pfText(std::string("            fov = ") + std::to_string(focalLength), false);
            //     pfText(std::string("eyes-height(cm) = ") + std::to_string(eyesHeight), false);
            //     pfText(std::string("camera-distance = ") + std::to_string(lookAtDistance), false);
            //     pfText(std::string(""), false);
            //     pfText(std::string("Multipliers:"), false);
            //     pfText(std::string("  position = ") + std::to_string(positionMultiplier), false);
            //     pfText(std::string("  rotation = ") + std::to_string(rotationMultiplier), false);
            //     pfText(std::string("    height = ") + std::to_string(heightMultiplier), false);
            //     pfText(std::string("  distance = ") + std::to_string(distanceMultiplier), false);
            //     pfText(std::string(""), false);
            // }
            // show delegates only if in selection mode
            // if (delegateSelectionMode)
            // {
            //     pfText(std::string("Available delegates:"), false);
            //     for (size_t i = 0; i < renderDelegates.size(); ++i)
            //     {
            //         std::string currline = "[" + std::to_string(i) + "] "
            //             + engine->GetRendererDisplayName(renderDelegates[i])
            //             + " (" + renderDelegates[i].GetString() + ")";
            //         pfText(currline, newDelegate == i);
            //     }
            // }
            // else
            // {
            //     std::string currline = engine->GetRendererDisplayName(renderDelegates[currentDelegate])
            //         + " (" + renderDelegates[currentDelegate].GetString() + ")";
            //     pfText(currline, false);
            // }
        }
        glPopMatrix();

        // Keep running
        glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();

	return 0;
}