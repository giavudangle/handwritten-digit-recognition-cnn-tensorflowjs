import * as tf from "@tensorflow/tfjs";


// Hàm inferences nhận vào 2 args : Mảng data,và data chưa convert
async function inference(imgArrayData, imgData) {
    const img2d = tf.tensor2d(imgArrayData, [28, 28]); // trả về 2d images với size là 28,28
    const imgToInference = img2d.reshape([1, 28, 28, 1]) // reshape về 4 chiều
    let img = tf.fromPixels(imgData, 1); 
    img = img.reshape([1, 28, 28, 1]); // reshape về 4 chiều
    img = tf.cast(img, 'float32');

    // Load model
    const loadedModel = await tf.loadModel('localstorage://saved-model');  // Nếu chưa có models phải train
    // Load model là core api từ tensorflow js

    if (loadedModel){
        const output = loadedModel.predict(imgToInference);
        const axis = 1;
        const predictions = Array.from(output.argMax(axis).dataSync());
        const labels = document.getElementsByClassName("number");
        for(let i=0; i< labels.length; i +=1 ) {
            labels[i].style.backgroundColor = "#fff";
        }
        const label = document.getElementById(`${predictions[0]}`);
        label.style.backgroundColor = "#00FF15";
    } else {
        alert('Không tìm thấy models bên trong LocalStorage, vui lòng train models trước khi predict');
    }
}

export default inference;