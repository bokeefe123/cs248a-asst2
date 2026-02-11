implementing "../renderer.slang";


float3 simpleShade(float3 rayDir, float3 normal, float3 albedo)
{
    float NdotL = max(dot(normalize(normal), normalize(-rayDir)), 0.0);
    return albedo * NdotL;
}

public struct RayMeshSampleResult
{
    public float t;
    public float3 normal;
    public float4 color;
}

public struct RayMeshIntersectionResult
{
    public RayHitResult closestHit;
    public Triangle hitTriangle;
}

public Optional<RayMeshSampleResult> meshSample(float2 uv, RendererUniform uniforms)
{
    // We generate a camera ray for the given uv coordinate.
    // You need to implement the `generateRay` function in `Camera` struct in `src/cs248a_renderer/slang_shaders/model/camera.slang`
    Ray ray = uniforms.camera.generateRay(uv);
    RayHitResult result;

    Optional<RayMeshSampleResult> allResult;
    // Ray mesh intersection test.
    Optional<RayMeshIntersectionResult> rayMeshIntersectionResult = rayMeshIntersection(ray, uniforms);
    if (rayMeshIntersectionResult.hasValue)
    {
        RayHitResult result = rayMeshIntersectionResult.value.closestHit;
        float3 hitPoint = ray.origin + result.t * ray.direction;
        float3 barycentricCoord = rayMeshIntersectionResult.value.hitTriangle.calculateBarycentricCoord(hitPoint);
        float3 normal = result.normal;
        float2 tex_uv = rayMeshIntersectionResult.value.hitTriangle.getUV(barycentricCoord);
        float4 color = shadeTriangle(uv, uniforms, rayMeshIntersectionResult.value.hitTriangle, ray, tex_uv, normal, barycentricCoord);
        allResult = Optional<RayMeshSampleResult>(RayMeshSampleResult(result.t, normal, color));
    }

    return allResult;
}

public Optional<RayMeshIntersectionResult> rayMeshIntersection(Ray ray, RendererUniform uniforms)
{
    Optional<RayHitResult> result = Optional<RayHitResult>();

    // IMPORTANT CHANGE: We need to store the hit triangle.
    Optional<RayMeshIntersectionResult> allResult = Optional<RayMeshIntersectionResult>();
    Optional<Triangle> hitTriangle = Optional<Triangle>();
    // IMPORTANT CHANGE: We need to store the hit triangle.

    // TODO: Student implementation starts here.
    Optional<RayHitResult> closestHitResult;
    Optional<RayHitResult> hitResult;
    if (uniforms.useBVH) {
        Optional<BVH<Triangle>.HitResult> bvhHitResult = uniforms.bvh.hit(ray);
        if (bvhHitResult.hasValue) {
            closestHitResult = Optional<RayHitResult>(bvhHitResult.value.rayHitResult);
            hitTriangle = Optional<Triangle>(bvhHitResult.value.primitive);
        }
    } else {
        for (uint i = 0; i < uniforms.triangleCount; i++)
        {
            Triangle triangle = uniforms.triangleBuf[i];
            hitResult = triangle.hit(ray);
            if (!hitResult.hasValue) {}
            else if (!closestHitResult.hasValue) {
                closestHitResult = hitResult;
                hitTriangle = Optional<Triangle>(triangle);
            }
            else if (hitResult.value.t < closestHitResult.value.t) {
                closestHitResult = hitResult;
                hitTriangle = Optional<Triangle>(triangle);
            }  
        }
    }
    result = closestHitResult;

    
    // TODO: Student implementation ends here.

    if (result.hasValue && hitTriangle.hasValue)
    {
        allResult = Optional<RayMeshIntersectionResult>(RayMeshIntersectionResult(result.value, hitTriangle.value));
    }

    return allResult;
}

// Helper function to get texture UV at an offset screen position
float2 getTexUVAtOffset(RendererUniform uniforms, Triangle triangle, float2 offsetUV)
{
    Ray ray = uniforms.camera.generateRay(offsetUV);
    Optional<RayHitResult> result = triangle.hit(ray);
    if (!result.hasValue) {
        return float2(0.0);
    }
    float3 hitPoint = ray.origin + result.value.t * ray.direction;
    float3 barycentricCoord = triangle.calculateBarycentricCoord(hitPoint);
    return triangle.getUV(barycentricCoord);
}

// Compute the level of the texture at the hit point.
// @param uv: The uv coordinates of the hit point.
// @param uniforms: The renderer uniforms.
// @param triangle: The triangle that was hit.
// @param tex_uv: The uv coordinates of the texture at the hit point.
// @return: The level of the texture at the hit point.
public float getLevel(float2 uv, RendererUniform uniforms, Triangle triangle, float2 tex_uv)
{
    float level = 0.0;

    // TODO: Student implementation starts here.
    
    // Compute pixel size in normalized screen coordinates
    float dx = 1.0 / float(uniforms.camera.canvasSize.x);
    float dy = 1.0 / float(uniforms.camera.canvasSize.y);
    
    // Ray differential method: shoot 4 rays (∂x⁺, ∂x⁻, ∂y⁺, ∂y⁻)
    // Use central differences for more accurate derivative estimation
    
    // Shoot rays offset in +x and -x direction to compute ρₓ
    float2 texUV_px_pos = getTexUVAtOffset(uniforms, triangle, uv + float2(dx, 0.0));
    float2 texUV_px_neg = getTexUVAtOffset(uniforms, triangle, uv - float2(dx, 0.0));
    float2 duvdx = (texUV_px_pos - texUV_px_neg) / (2.0 * dx);
    float rho_x = length(duvdx);
    
    // Shoot rays offset in +y and -y direction to compute ρᵧ
    float2 texUV_py_pos = getTexUVAtOffset(uniforms, triangle, uv + float2(0.0, dy));
    float2 texUV_py_neg = getTexUVAtOffset(uniforms, triangle, uv - float2(0.0, dy));
    float2 duvdy = (texUV_py_pos - texUV_py_neg) / (2.0 * dy);
    float rho_y = length(duvdy);
    
    // D = log₂(max(ρₓ, ρᵧ))
    float rho = max(rho_x, rho_y);
    level = log2(max(rho, 1e-6));  // Avoid log2(0)
    
    // TODO: Student implementation ends here.
    return level;
}   

public float4 shadeTriangle(float2 uv, RendererUniform uniforms, Triangle triangle, Ray ray, float2 tex_uv, float3 normal, float3 barycentricCoord)
{
    if (uniforms.visualizeBarycentricCoords)
    {
        return float4(barycentricCoord, 1.0);
    }
    if (uniforms.visualizeTexUV)
    {
        return float4(tex_uv, 0.0, 1.0);
    }
    // Get the level of detail of the texture at the hit point.
    float level = getLevel(uv, uniforms, triangle, tex_uv);
    uint discreteLevel = uint(level);

    // Sample albedo from the material at the hit point.
    uint materialId = triangle.materialId + triangle.offset;
    PhysicsBasedMaterial physicsBasedMaterial = uniforms.physicsBasedMaterialBuf[materialId];
    PhysicsBasedParameters parameters = sampleMaterial(physicsBasedMaterial, uniforms.physicsBasedMaterialTextureBuf, tex_uv, level, uniforms.visualizeLevelOfDetail);
    float3 albedo = parameters.albedo;

    if (uniforms.visualizeLevelOfDetail || uniforms.visualizeAlbedo)
    {
        return float4(float3(albedo), 1.0);
    }

    // Shade the triangle.
    float3 color = simpleShade(ray.direction, normal, albedo);
    return float4(color, 1.0);
}
