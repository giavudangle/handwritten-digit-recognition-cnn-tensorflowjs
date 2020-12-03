import * as tf from '@tensorflow/tfjs';

class GenerateData {
    // Khởi tạo constructor
    constructor(props) {
        this.FEATURES_LINK = "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png"; // bộ data lấy từ MNIST trên googleapis
        this.LABELS_LINK = "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8"; // // bộ labels lấy từ MNIST trên googleapis
        this.CLASSES_NUM = 10; // Classes 10 : 0 - 9
        this.TOTAL_DATASET_NUM = 65000; // Tổng dataset là 65000
        this.TRAIN_NUM = 55000; // Tập train: 55000
        this.TEST_NUM = 10000;  // Tập test: 10000
        this.IMG_SIZE = 784; // Size hình
        this.CHUNK_SIZE = 5000; // Phân cụm nhỏ 
        this.currentTrainIndex = 0;
        this.currentTestIndex = 0;
    }
    
    /**
     * 1. Trích xuất tính chất đặc trưng của ảnh hàm : extractFeatures() 
     */
    extractFeatures() {
        return new Promise((resolve, reject) => {
            const img = new Image(); // Tạo 1 image từ HTML
            img.crossOrigin = '';
            img.src = this.FEATURES_LINK; // Source lấy từ LINK MNIST
            
            // Tạo canvas để có thể vẽ
            const cv = document.createElement('canvas');
            const ctx = cv.getContext('2d');
            img.onload = () => {
                img.height = img.naturalHeight; // 65000
                img.width = img.naturalWidth; // 784
                // Vậy canvas có chiều dài là -> 650000 * 5000
                cv.width = img.width;
                cv.height = this.CHUNK_SIZE;

                
                // 4 ở đây là 4 kênh màu [0, 0, 0, 255]
                const datasetBuff = new ArrayBuffer(this.TOTAL_DATASET_NUM * this.IMG_SIZE * 4) // 65000 * 784 * 4
                
                // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/DataView
                // Data View
                for(let i=0; i < this.TOTAL_DATASET_NUM/this.CHUNK_SIZE; i+=1) { // 0 -> 12 
                    // Float32Array(3920000)
                    // Tạo dataview để hold những pixel
                    // Set chunk = 5000 ->
                    // 65000 / 5000 = 13 dataview
                    const datasetBytesView = new Float32Array(datasetBuff, 
                        i * this.CHUNK_SIZE * this.IMG_SIZE * 4
                        , this.IMG_SIZE * this.CHUNK_SIZE);
                    
                    // https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/drawImage
                    // Dataset từ MNIST là 1 ảnh lớn có size là: 784 * 65000 (width * height)
                    // Tạo thanh trượt với size 784 * 5000 - trượt từ trên xuống dưới
                    // -> Vậy chúng ta có size 784 * 5000
                    
                    ctx.drawImage(img, 0, i * this.CHUNK_SIZE, img.width, this.CHUNK_SIZE, 0, 0, img.width, this.CHUNK_SIZE);
                    const imgData = ctx.getImageData(0, 0, img.width, this.CHUNK_SIZE);
                    const imgDateLength = imgData.data.length

                    // Lặp qua mỗi pixel trên ảnh
                    for(let j = 0; j < imgDateLength / 4; j +=1) { 
                        // All channel has same value -> only need to read red channel
                        // Nếu tất cả các kênh màu đều có cùng giá trị -> chỉ cần độc kênh màu đỏ
                        const red_index = j * 4;
                        // Map mảng về mảng đơn vị
                        datasetBytesView[j] = imgData.data[red_index] / 255;
                    }
                    console.log('Done Extracting Labels for chunk: ', i);
                }

                // 784 * 65000 
                //-> 784: Số pixel ảnh đã được làm phẳng
                //-> 65000: Số image
                // Mỗi dòng biểu diễn cho một bức ảnh có size 28*28
                // Mỗi phần tử giữ 1 pixel dữ liệu
           
                this.datasetImgs = new Float32Array(datasetBuff); 
                resolve(); // giải quyết sau khi xử lí xong ( api của web đi cùng với Promies)
            }
        })
    }

    /**
     * 1. Trích xuất tính chất đặc trưng labels : extractLabels() 
     */
    extractLabels() {
        // Xử lý bất đồng bộ với Promise
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


    load() {
        return new Promise((resolve, reject) => {
            const promises = [this.extractFeatures(), this.extractLabels()]
            Promise.all(promises).then(() => {
                console.log("Finish extract datas and labels");
                // Generate shuffled train and test indicies
                // Uint32Array(55000) with shuffled indicies
                this.trainIndicies = tf.util.createShuffledIndices(this.TRAIN_NUM);
                this.testIndicies = tf.util.createShuffledIndices(this.TEST_NUM);


                // Generate train and test images
                this.trainImgs = this.datasetImgs.slice(0, this.TRAIN_NUM * this.IMG_SIZE);
                this.testImgs = this.datasetImgs.slice(this.TRAIN_NUM * this.IMG_SIZE);

                // Generate train and test labels
                this.trainLabels = this.labels.slice(0, this.TRAIN_NUM * this.CLASSES_NUM);
                this.testLabels = this.labels.slice(this.TRAIN_NUM * this.CLASSES_NUM);
                resolve();
            }).catch(err => {
                reject(err);
            });
        });
    }

    nextBatch(type, batchSize) {
        let images;
        let labels;
        const batchImgs = new Float32Array(this.IMG_SIZE * batchSize);
        const batchLabels = new Uint8Array(this.CLASSES_NUM * batchSize);
        let idx;
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

        for(let i =0; i < batchSize; i += 1) {
            const index = idx[i];
            const image = images.slice(index * this.IMG_SIZE, (index+1) * this.IMG_SIZE)
            const label = labels.slice(index * this.CLASSES_NUM, (index + 1) * this.CLASSES_NUM)
            batchImgs.set(image, i * this.IMG_SIZE);
            batchLabels.set(label, i * this.CLASSES_NUM);
        }

        return {
            images: tf.tensor2d(batchImgs, [ batchSize, this.IMG_SIZE ]),
            labels: tf.tensor2d(batchLabels, [ batchSize, this.CLASSES_NUM ])
        }
    }
}

export default GenerateData;