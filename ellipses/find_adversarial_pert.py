import tensorflow as tf
from operators import proj_l2_ball


def find_adv_pert(opA, model, images, optimizer, lam=0.0, magn_initial_noise=1e-3, max_nbr_epochs=200,  proj_ell2_ball=False, radius=None, dtype=tf.float32, verbose=1):

    batch_size = images.shape[0];

    measurements = tf.cast(opA(images, output_real=True), dtype=dtype)
    radius = tf.cast(radius, dtype=dtype) 

    x0 = opA.adjoint(measurements, output_real=True)

    orig_rec = model(x0, training=False);

    norm_measurements = tf.cast(tf.norm(measurements), dtype) 
    norm_orig_rec = tf.norm(orig_rec) 

    if radius is not None and verbose > 0:
        print('radius: ', radius)
    
    ee = tf.Variable(initial_value=magn_initial_noise*tf.random.uniform(measurements.shape, dtype=dtype));

    @tf.function
    def attack_step():
        with tf.GradientTape() as tape:
            me = opA.adjoint(measurements+ee, output_real=True);
            prediction = model(me, training=False)
            loss_pred = tf.nn.l2_loss(prediction-orig_rec)
            loss_ee = tf.nn.l2_loss(ee); 
            loss = -loss_pred + lam*loss_ee
            gradients = tape.gradient(loss, [ee])
        optimizer.apply_gradients(zip(gradients, [ee]))
        if proj_ell2_ball:
            s = proj_l2_ball(ee, radius)
            ee.assign(s)
        diff_pred = tf.math.sqrt(tf.reduce_sum(tf.math.pow(tf.math.abs(prediction-orig_rec),2)))
        norm_ee = tf.math.sqrt(tf.reduce_sum(tf.math.pow(tf.math.abs(ee), 2)))

        return diff_pred, norm_ee


    epoch = 1
    norm_ee = 0;
    if proj_ell2_ball:
        condition = lambda epoch, norm_ee: epoch <=max_nbr_epochs
    else:
        condition = lambda epoch, norm_ee: norm_ee <= radius 

    while condition(epoch, norm_ee):

        norm_diff_pred, norm_ee = attack_step()

        rel_err_pred = norm_diff_pred/norm_orig_rec;
        rel_err_noise = norm_ee/norm_measurements;

        str1 = f"i: {epoch:03}, |f(y+e) - f(y)|: {norm_diff_pred:6f}, |e|: {norm_ee:6f}, "
        str2 = f"rel_err_rec: {rel_err_pred:6f}, rel_err_noise: {rel_err_noise:6f}"
        if verbose > 0:
            print(str1+str2)

        epoch += 1        

    rr = opA.adjoint(ee)
    ee = opA._to_complex(ee);

    return ee.numpy(), rr.numpy();





