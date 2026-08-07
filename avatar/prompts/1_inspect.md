# Prompt 1 — inspect the model (send this FIRST)

Do not rig anything yet. We do not know what state this FBX is in, and the
right approach differs completely depending on what is already there.

---

I have an FBX character model I want to drive with motion capture data derived
from MediaPipe landmarks. Before we change anything, inspect it and report back.

Import the FBX and tell me:

1. **Does it have an armature at all?** If not, say so plainly — that changes
   the plan entirely.

2. **Full bone list**, with hierarchy. I specifically need to know:
   - spine / chest / neck / head chain
   - shoulder, upper arm, forearm, hand for both sides
   - whether there are FINGER bones, and how many per finger
     (MediaPipe gives 4 joints per finger, so 3 bones per finger is ideal)

3. **Naming convention** — is it Mixamo (`mixamorig:LeftArm`), Rigify
   (`upper_arm.L`), Unreal, or custom?

4. **Rest pose** — T-pose, A-pose, or something else? Which axis is up, and
   which way does the character face?

5. **Skinning** — is the mesh already weighted to the armature, or is it a
   static mesh sitting next to a skeleton?

6. **Scale and units** — how tall is the character in Blender units?

7. Any existing animations / Actions already on it?

Report all of that before making any modifications. I will decide the
retargeting approach based on what you find.
