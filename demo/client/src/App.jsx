import {
  Card,
  Slider,
  TextArea,
  TextField,
  Heading,
  Button,
} from "@radix-ui/themes";
import { useState, useRef, useEffect } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const points = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

function App() {
  const canvasRef = useRef();
  const cameraRef = useRef();
  const rendererRef = useRef();
  const [prompt, setPrompt] = useState("");

  useEffect(() => {
    const canvas = canvasRef.current;
    // const ctx = canvas.getContext("2d");

    // ctx.fillStyle = "#111111";

    const renderer = new THREE.WebGLRenderer({ canvas });

    const scene = new THREE.Scene();

    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    camera.position.z = 5;
    cameraRef.current = camera;
    rendererRef.current = renderer;

    const geometry = new THREE.BufferGeometry();
    const vertices = [];

    points.forEach((point) => {
      vertices.push(point[0], point[1], point[2]);
    });

    geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(vertices, 3)
    );

    const sphereGeometry = new THREE.SphereGeometry(0.05, 32, 32);
    const material = new THREE.MeshPhongMaterial({
      color: "#fff",
    });

    points.forEach(([x, y, z]) => {
      const sphere = new THREE.Mesh(sphereGeometry, material);
      sphere.position.set(x, y, z);
      scene.add(sphere);
    });

    const controls = new OrbitControls(camera, renderer.domElement);

    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(1, 1, 1).normalize();
    scene.add(directionalLight);

    scene.background = new THREE.Color("#111111");

    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      const camera = cameraRef.current;
      const renderer = rendererRef.current;

      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener("resize", handleResize);

    return () => {
      renderer.dispose();
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("handle submit");
  };

  return (
    <div style={{ height: "100vh", position: "relative" }}>
      <Heading
        style={{
          position: "absolute",
          top: 64,
          left: "50%",
          transform: "translateX(-50%)",
        }}
        size="4"
      >
        {prompt}
      </Heading>

      <canvas ref={canvasRef} />

      <form
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          position: "absolute",
          width: 512,
          bottom: 64,
          left: "50%",
          transform: "translateX(-50%)",
        }}
        onSubmit={handleSubmit}
      >
        <TextField.Root
          style={{ flex: 1, marginRight: 8 }}
          placeholder="Create your 3D object ..."
          variant="soft"
          size="3"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />

        <Button size="3" highContrast={!!prompt} disabled={!prompt}>
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            color={prompt ? "black" : "white"}
          >
            <path
              d="M7.14645 2.14645C7.34171 1.95118 7.65829 1.95118 7.85355 2.14645L11.8536 6.14645C12.0488 6.34171 12.0488 6.65829 11.8536 6.85355C11.6583 7.04882 11.3417 7.04882 11.1464 6.85355L8 3.70711L8 12.5C8 12.7761 7.77614 13 7.5 13C7.22386 13 7 12.7761 7 12.5L7 3.70711L3.85355 6.85355C3.65829 7.04882 3.34171 7.04882 3.14645 6.85355C2.95118 6.65829 2.95118 6.34171 3.14645 6.14645L7.14645 2.14645Z"
              fill="currentColor"
              fill-rule="evenodd"
              clip-rule="evenodd"
            ></path>
          </svg>
        </Button>
      </form>
    </div>
  );
}

export default App;
