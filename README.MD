### Training, Inference Convolutional Neurel Network 

| Resource | Link |
| ------ | ------ |
| Features | [MNIST Data Features](https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png) |
| Labels | [MNIST Data Labels](https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8) |


<img src="https://lh3.googleusercontent.com/9QHeeMU2xKoO7LSzuQ3jqDsTgYMgFEHImUe1o2Xkk_CwKPSg4dv878S1xcPAIAg_XgaFrnwQ-njiryOIAUm0KNY2G4BNNW_sEqe4rootS4UqWBtKLMN2VScXLlR0W6c7KdLKiDaNtnQ" width=500>


Tạo mạng CNN với 2 lớp ẩn

  - **Layer Convolution**
    - input **[28, 28, 1]** // đầu vào
    - kernel size: **5** // cho trượt trên kernel có size là 5
    - activation Function: **reLu** // hàm kích hoạt dùng Relu
    - number of Filters: **8** // 8 bộ lọc
    - strides: **1**  // bước chạy là 1
  - **Layer MaxPooling**
      - pool Size: **2**
      - strides: **2**
  - **Last Layer**
      - Softmax Layer with **10 classes**

<img src="https://codetolight.files.wordpress.com/2017/11/network.png?w=1024" width=500>
### Installation and Run

```sh
$ cd ./
$ yarn
$ yarn dev
```

```sh
Open 127.0.0.1:1234
```

```
If you haven't trained, please run Train button.
```

### Training

![](https://lh3.googleusercontent.com/wfQUynof8T4CXJIPV0FvQvbJHwK9zXeGKm5QmzJq_0A-bqYKzqz4fFRwicGBgPK4-xKRYR1VQFy8WpLvDVWm6t-4bWSi5RxWmzZD1zKDQT3jUfxVszhaNEeJjEbD6XwQ_sBQIWrizXs)



## Data Processing
### Input Data
Matrix 6500 * 784
= 6500 * 28 * 28
= 4,738,500
### Extract Data
```c
  for(let i=0; i < this.TOTAL_DATASET_NUM/this.CHUNK_SIZE; i+=1) { // 0 -> 12
                    // Float32Array(3920000)
                    // Create data view to hold value of each pixel.
                    // We will have 13 data view
                    const datasetBytesView = new Float32Array(datasetBuff, 
                        i * this.CHUNK_SIZE * this.IMG_SIZE * 4
                        , this.IMG_SIZE * this.CHUNK_SIZE);
                    
                    // https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/drawImage
                    // Dataset is a big image with size 784 * 65000 (width * height)
                    // Create a slider with size 784 * 5000 and slide from top to bottom 
                    // -> we will get 13 chunk with size 784 * 5000
                    
                    ctx.drawImage(img, 0, i * this.CHUNK_SIZE, img.width, this.CHUNK_SIZE, 0, 0, img.width, this.CHUNK_SIZE);
                    const imgData = ctx.getImageData(0, 0, img.width, this.CHUNK_SIZE);
                    const imgDateLength = imgData.data.length

                    // Loop through each pixel in chunk
                    for(let j = 0; j < imgDateLength / 4; j +=1) { 
                        // All channel has same value -> only need to read red channel
                        const red_index = j * 4;
                        // Nomarlize pixel value to [0, 1]
                        datasetBytesView[j] = imgData.data[red_index] / 255;
                    }
                    console.log('Done Extracting Labels for chunk: ', i);
                }
                // 784 * 65000 -> 784: flattended image pixels, 65000: number of images
                // each row represent an img 28 * 28
                // each element hold one nomalized pixel data
                this.datasetImgs = new Float32Array(datasetBuff); 
                resolve();
            }
```

### Extract Labels
```c
    extractLabels() {
        return new Promise((resolve, reject) => {
            fetch(this.LABELS_LINK).then(res => {
                res.arrayBuffer().then(buff => {
                    const labels = new Uint8Array(buff);
                    this.labels = labels;
                    console.log(labels);
                    resolve();
                }).catch(err => reject(err))
            }).catch(err => reject(err))
        })
    }
```

### Stochastic Gradient Descent Next Batch (Optimize)

```c
if(type === "train") {
      [ images, labels ] = [ this.trainImgs, this.trainLabels ];
      const newTrainIndex = this.currentTrainIndex + batchSize;
      idx = this.trainIndicies.slice(this.currentTrainIndex, newTrainIndex);
      this.currentTrainIndex = newTrainIndex;
  } else if (type === "test") {
      [ images, labels ] = [ this.testImgs, this.testLabels ];
      const newTestIndex = this.currentTestIndex + batchSize;
      idx = this.trainIndicies.slice(this.currentTestIndex, newTestIndex);
      this.currentTestIndex = newTestIndex;
  }
```

### Convert to Tensor
```c
return {
    images: tf.tensor2d(batchImgs, [ batchSize, this.IMG_SIZE ]),
    labels: tf.tensor2d(batchLabels, [ batchSize, this.CLASSES_NUM ])
}
```

### CNN - Setup Sequential Network
```c
// Implement Sequential Neural Network
const model = tf.sequential()
```