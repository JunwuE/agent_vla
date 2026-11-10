def run(ctx):
    import numpy as np
    import json
    from scipy.spatial.transform import Rotation as Rot

    class Halt(Exception):
        pass
    class Done(Exception):
        pass

    used = 0
    def dec(v):
        return json.loads(v) if isinstance(v, str) else v

    def observe():
        return dec(ctx.observe())

    def success():
        if ctx.check('task_success')['passed'] is True:
            raise Done()

    def fail(message):
        ctx.note(facts=[message], hypotheses=['Whole-task completion is unresolved; code retains ownership.'])
        raise Halt(message)

    def record_cloud(camera):
        record = ctx.read_observation(camera)
        meta = record['metadata']
        cal = meta.get('calibration', {})
        if cal.get('alignment_verified') is not True or cal.get('frame') != 'optical_to_osc_arm_base':
            fail('Missing verified RGB-D calibration.')
        if cal.get('step') != meta['step'] or 'depth_m' not in record:
            fail('Depth/calibration snapshot mismatch.')
        depth = np.asarray(record['depth_m'], dtype=float)
        rgb = np.asarray(record['rgb'])
        if rgb.shape[:2] != depth.shape:
            fail('RGB and depth shape mismatch.')
        k = np.asarray(cal['intrinsics'], dtype=float)
        transform = np.asarray(cal['camera_to_base'], dtype=float)
        vv, uu = np.indices(depth.shape)
        optical = np.stack([(uu-k[0,2])*depth/k[0,0],
                            (vv-k[1,2])*depth/k[1,1], depth], axis=-1)
        cloud = optical @ transform[:3,:3].T + transform[:3,3]
        valid = np.isfinite(depth) & (depth > 0.05) & (depth < 3.0)
        gray = rgb[...,:3].astype(float).mean(axis=-1)
        neutral = np.ptp(rgb[...,:3].astype(float), axis=-1) < 12
        metal = valid & neutral & (gray > 55) & (gray < 205)
        return record, cloud, valid, metal, uu, vv

    def grounded(camera, box, label):
        obs = observe()
        g = dec(ctx.ground_target(target=label, camera=camera,
                observation_step=obs['step'], bbox=[int(x) for x in box], confidence=0.8))
        m = dec(ctx.query_geometry(g['grounding_id']))
        if m.get('reliable') is not True or m.get('frame') != 'osc_arm_base':
            fail('Selected rim has unreliable metric geometry: ' + label)
        if m.get('depth_iqr_m', 1) > 0.045:
            fail('Selected rim has excessive depth dispersion: ' + label)
        return np.asarray(m['centroid_m'], dtype=float)

    def move_to(position, orientation, closed, max_calls, label, tolerance=0.007):
        nonlocal used
        position = np.asarray(position, dtype=float)
        desired = Rot.from_quat(orientation)
        stalled = 0
        with ctx.stage(label) as stage:
            for _ in range(max_calls):
                success()
                obs = observe()
                pose = obs['eef_control_pose']
                p = np.asarray(pose['position'], dtype=float)
                current = Rot.from_quat(pose['quaternion_xyzw'])
                delta = position-p
                dp = float(np.linalg.norm(delta))
                rv = (desired * current.inv()).as_rotvec()
                dr = float(np.linalg.norm(rv))
                if dp < tolerance and dr < 0.07:
                    stage.record(observation_step=obs['step'], position_error_m=dp, rotation_error_rad=dr)
                    return
                if used+5 > 240:
                    fail('This revision reached its 240-action bound.')
                step_delta = delta*min(1.0, 0.038/max(dp,1e-9))
                rotation_delta = rv*min(1.0,0.25/max(dr,1e-9))
                target = {'position':(p+step_delta).tolist(),
                          'quaternion_xyzw':(Rot.from_rotvec(rotation_delta)*current).as_quat().tolist(),
                          'gripper_closed':bool(closed), 'base_motion':[0.,0.,0.,0.],
                          'control_mode':'arm'}
                ctx.move_eef(target, steps=5)
                used += 5
                after = observe()
                pa = np.asarray(after['eef_control_pose']['position'])
                qa = Rot.from_quat(after['eef_control_pose']['quaternion_xyzw'])
                progress = np.linalg.norm(pa-p) + 0.1*np.linalg.norm((qa*current.inv()).as_rotvec())
                stalled = stalled+1 if progress < 0.0015 else 0
                stage.record(observation_step=after['step'], measured_position=pa.tolist(),
                             position_error_m=float(np.linalg.norm(position-pa)))
                if stalled >= 2:
                    fail('Two feedback segments produced negligible measured progress during '+label)
            pose = observe()['eef_control_pose']
            dp = np.linalg.norm(position-np.asarray(pose['position']))
            dr = np.linalg.norm((desired*Rot.from_quat(pose['quaternion_xyzw']).inv()).as_rotvec())
            if dp > 0.012 or dr > 0.12:
                fail('Bounded pose tracking did not converge during '+label)
    
    def gripper(closed):
        nonlocal used
        if used+5 > 240:
            fail('Action bound reached before gripper change.')
        ctx.set_gripper(closed, steps=5)
        used += 5
        success()

    try:
        obs = observe()
        success()
        if obs['step'] != 1735:
            fail('The initial selected regions require the step-1735 image.')
        with ctx.stage('fresh rim and hinge surface measurement') as stage:
            rim = grounded('right',[61,179,77,183],'central outer door rim')
            hinge = grounded('right',[73,152,94,156],'hinge-side rim')
            corner = grounded('right',[80,175,85,180],'independent right-side rim tracking patch')
            rec, cloud, valid, metal, uu, vv = record_cloud('right')
            patch = cloud[179:183,61:77]
            valid_patch = valid[179:183,61:77]
            calibrated_median = np.median(patch[valid_patch],axis=0)
            if np.linalg.norm(calibrated_median-rim) > 0.02:
                fail('Raw calibrated cloud disagrees with grounded geometry.')
            up = np.array([0.,0.,1.])
            outward = rim-hinge
            outward[2] = 0
            radius = np.linalg.norm(outward)
            if not 0.14 < radius < 0.30:
                fail('Measured door length is implausible for this selection.')
            outward /= radius
            inward = -outward
            axis = np.cross(outward,up)
            axis /= np.linalg.norm(axis)
            lateral = axis
            base_rotation = Rot.from_matrix(np.column_stack([np.cross(up,inward),up,inward]))
            # A surface-based hinge estimate; not privileged joint state.
            hinge = hinge.copy()
            hinge[2] = rim[2]
            radius_vector = rim-hinge
            side_coordinate = float(np.dot(corner-hinge,lateral))
            side_radius = float(np.linalg.norm(corner-hinge))
            stage.record(rim=rim.tolist(),hinge=hinge.tolist(),corner=corner.tolist(),
                         estimated_radius_m=float(radius),side_radius_m=side_radius)

        def fresh_rim(estimate):
            rec, cloud, valid, metal, uu, vv = record_cloud('right')
            distance = np.linalg.norm(cloud-estimate,axis=-1)
            mask = metal & (distance < 0.023)
            if np.count_nonzero(mask) < 5:
                fail('The outer rim cannot be reselected from the current unobscured RGB-D view.')
            p = np.median(cloud[mask],axis=0)
            xs,ys = uu[mask],vv[mask]
            box=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
            g = dec(ctx.ground_target(target='freshly reselected outer rim surface',
                     camera='right',observation_step=rec['metadata']['step'],bbox=box,confidence=0.7))
            check = dec(ctx.query_geometry(g['grounding_id']))
            if check.get('reliable') is not True:
                fail('Reselected rim geometry is unreliable.')
            if np.linalg.norm(p-estimate) > 0.018:
                fail('Rim moved or is occluded before the planned grasp.')
            return p

        def track_angle(previous, expected):
            observations=[]
            for camera in ['right','left']:
                rec, cloud, valid, metal, uu, vv = record_cloud(camera)
                offset=cloud-hinge
                horizontal=offset @ outward
                height=offset[...,2]
                transverse=offset @ lateral
                rho=np.sqrt(horizontal**2+height**2)
                angle=np.arctan2(height,horizontal)
                mask = (metal & (np.abs(transverse-side_coordinate)<0.014)
                        & (np.abs(rho-side_radius)<0.025)
                        & (angle>max(-0.10,previous-0.14))
                        & (angle<min(1.68,expected+0.25)))
                if np.count_nonzero(mask)<5:
                    continue
                angles=angle[mask]
                median=float(np.median(angles))
                dispersion=float(np.percentile(angles,75)-np.percentile(angles,25))
                if dispersion>0.16:
                    continue
                tight=mask & (np.abs(angle-median)<0.08)
                if np.count_nonzero(tight)<4:
                    continue
                xs,ys=uu[tight],vv[tight]
                box=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
                g=dec(ctx.ground_target(target='metric and appearance-selected door side rim',
                      camera=camera,observation_step=rec['metadata']['step'],bbox=box,confidence=0.65))
                m=dec(ctx.query_geometry(g['grounding_id']))
                if m.get('reliable') is not True:
                    continue
                observations.append(median)
            if not observations:
                fail('Door-side depth tracking is unavailable or ambiguous; no further closing motion.')
            if max(observations)-min(observations)>0.18:
                fail('The two camera estimates of door rotation disagree.')
            measured=float(np.median(observations))
            ctx.note(facts=['Measured side-rim angle estimate: '+str(measured)+
                           ' rad at step '+str(observe()['step'])],
                     hypotheses=['This visible-surface estimate supports contact monitoring but is not a completion verifier.'])
            return measured

        initial_angle=track_angle(0.0,0.0)
        p0=np.asarray(obs['eef_control_pose']['position'])
        q0=obs['eef_control_pose']['quaternion_xyzw']
        move_to(p0+0.065*outward+0.055*up,q0,False,5,'clear inner opening above door')
        q=base_rotation.as_quat()
        move_to(rim+0.075*outward+0.115*up,q,False,12,'align outside outer rim with vertical jaw separation')
        rim=fresh_rim(rim)
        hinge[2]=rim[2]
        radius_vector=rim-hinge
        # Insert the fingertip site slightly beyond the rim edge, near its mid-thickness.
        grip_point=rim-0.006*outward-0.006*up
        move_to(rim+0.055*outward-0.006*up,q,False,7,'lower outside the rim')
        rim=fresh_rim(rim)
        grip_point=rim-0.006*outward-0.006*up
        move_to(grip_point,q,False,5,'approach rim between open fingers',tolerance=0.005)
        gripper(True)
        gripper(True)
        obs=observe()
        aperture=abs(float(obs['state'][-2])-float(obs['state'][-1]))
        ctx.note(facts=['Measured finger separation after closure: '+str(aperture)],
                 hypotheses=['A nonzero finger separation alone does not establish a grasp.'])
        if aperture<0.003 or aperture>0.065:
            fail('Finger separation does not support continuing this rim-grasp experiment.')
        grip_start=np.asarray(obs['eef_control_pose']['position'])
        relative=grip_start-hinge
        measured_angle=initial_angle
        for requested_angle in [0.20,0.34,0.48,0.62,0.76,0.90,1.04,1.18,1.32,1.46,1.55]:
            rotation=Rot.from_rotvec(axis*requested_angle)
            target=hinge+rotation.apply(relative)
            target_q=(rotation*base_rotation).as_quat()
            move_to(target,target_q,True,4,'grasped closing arc '+str(requested_angle),tolerance=0.007)
            success()
            observed_angle=track_angle(measured_angle,requested_angle)
            if requested_angle==0.20 and observed_angle-initial_angle<0.07:
                fail('The initial lift did not produce measured door rotation; stop before further arc motion.')
            if requested_angle-observed_angle>0.22 or observed_angle<measured_angle-0.09:
                fail('Door-side movement no longer follows the closing arc; possible lost contact.')
            measured_angle=observed_angle
        with ctx.stage('verify complete door closure', completion_condition='task_success'):
            success()
        return 'SUBTASK_DONE'
    except Done:
        return 'SUBTASK_DONE'
    except Halt:
        return 'UNRESOLVED'
