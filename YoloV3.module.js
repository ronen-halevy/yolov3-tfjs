const configRender = {
  font: '20px serif',
  lineWidth: 3,
  lineColor: 'yellow',
  textColor: 'blue',
  textBackgoundColor: 'white',
};
// tf.setBackend('webgl');

class YoloV3 {
  constructor(model, anchors, nClasses, scoreTHR, iouTHR, maxBoxes) {
    this.model = model;
    this.anchors = anchors;
    this.nClasses = nClasses;
    this.scoreTHR = scoreTHR;
    this.iouTHR = iouTHR;
    this.maxBoxes = maxBoxes;
  }

  setScoreTHR = (val) => {
    this.scoreTHR = val;
  };
  setIouTHR = (val) => {
    this.iouTHR = val;
  };

  setMaxBoxes = (val) => {
    this.maxBoxes = val;
  };
  setModelParams(model, anchors, nClasses) {
    this.model = model;
    this.anchors = anchors;
    this.nClasses = nClasses;
  }

  imagePreprocess = (image) => {
    const imgTensor = tf.browser.fromPixels(image);
    // Yolo input width:
    const imageHeight = 416;
    const imageWidth = 416;
    var resized = tf.image.resizeBilinear(imgTensor, [imageHeight, imageWidth]);
    var tensor = resized.expandDims(0).toFloat();
    tensor = tensor.div(255);
    return tensor;
  };

  detectFrame = (imageFrame) => {
    tf.engine().startScope();

    const imageTensor = this.imagePreprocess(imageFrame);

    const modelOutputGrids = this.model.predict(imageTensor);

    // Decode predictions: combines all grids detection results
    let [bboxes, confidences, classProbs] = decode(
      modelOutputGrids,
      80,
      this.anchors
    );
    let axis = -1;
    let classIndices = classProbs.argMax(axis);
    classProbs = classProbs.max(axis);
    confidences = confidences.squeeze(axis);
    let scores = confidences.mul(classProbs);
    // clean mem
    classProbs.dispose();
    confidences.dispose();

    const nmsPromise = nms(
      bboxes,
      scores,
      classIndices,
      this.iouTHR,
      this.scoreTHR,
      this.maxBoxes
    ).then((reasultArrays) => {
      tf.engine().endScope();

      return reasultArrays;
    });
    return nmsPromise;
  };
}

const nms = (bboxes, scores, classIndices, iouTHR, scoreTHR, maxBoxes) => {
  const nmsPromise = new Promise((resolve) => {
    const nmsResults = tf.image.nonMaxSuppressionAsync(
      bboxes,
      scores,
      maxBoxes,
      iouTHR,
      scoreTHR
    );
    resolve(nmsResults);
  }).then((nmsResults) => {
    let selectedBboxes = bboxes.gather(nmsResults);
    let selectedClasses = classIndices.gather(nmsResults);
    let selectedScores = scores.gather(nmsResults);

    const bboxesArray = selectedBboxes.array();
    const scoresArray = selectedScores.array();
    const classIndicesArray = selectedClasses.array();
    let reasultArrays = Promise.all([
      bboxesArray,
      scoresArray,
      classIndicesArray,
    ]);

    selectedBboxes.dispose();
    selectedClasses.dispose();
    selectedScores.dispose();
    return reasultArrays;
  });

  return nmsPromise;
};

function decode(gridsOutputs, nClasses, anchors) {
  // for both yolo and tiny:
  const nanchors_per_scale = 3; // Large, Medium, Small
  const anchor_entry_size = 2; // width, height

  let anchors_table = tf.reshape(anchors, [
    -1,
    nanchors_per_scale,
    anchor_entry_size,
  ]);

  let bboxes = [];
  let confidences = [];
  let classProbs = [];
  for (let idx = 0; idx < gridsOutputs.length; idx++) {
    let axis = -1;
    let [xy, wh, obj, class_prob] = tf.split(
      gridsOutputs[idx],
      [2, 2, 1, nClasses],
      axis
    );
    let anchors = tf.slice(anchors_table, [idx], 1);
    const bboxes_in_grid = arrange_bbox(tf.sigmoid(xy), wh.exp().mul(anchors));

    bboxes.push(
      tf.reshape(bboxes_in_grid, [
        bboxes_in_grid.shape[0],
        -1,
        bboxes_in_grid.shape[4],
      ])
    );

    confidences.push(
      tf.reshape(tf.sigmoid(obj), [obj.shape[0], -1, obj.shape[4]])
    );
    classProbs.push(
      tf.reshape(tf.sigmoid(class_prob), [
        class_prob.shape[0],
        -1,
        class_prob.shape[4],
      ])
    );
  }

  let axis = 1;
  bboxes = tf.concat(bboxes, axis);
  confidences = tf.concat(confidences, axis);
  classProbs = tf.concat(classProbs, axis);

  axis = 0;
  bboxes = bboxes.squeeze(axis);
  classProbs = classProbs.squeeze(axis);
  confidences = confidences.squeeze(axis);

  return [bboxes, confidences, classProbs];
}

function arrange_bbox(xy, wh) {
  let grid_size = [xy.shape[1], xy.shape[1]];

  let grid = tf.meshgrid(
    tf.range(0, xy.shape[1], 1),
    tf.range(0, xy.shape[1], 1)
  );
  var axis = -1;
  grid = tf.stack(grid, axis);

  axis = 2;
  grid = grid.expandDims(axis);

  xy = xy.add(tf.cast(grid, 'float32'));
  xy = xy.div(tf.cast(grid_size, 'float32'));

  let value1 = tf.scalar(2);
  wh = wh.div(value1);
  var xy_min = xy.sub(wh);
  var xy_max = xy.add(wh);

  var bbox = tf.concat([xy_min, xy_max], -1);
  grid.dispose();
  grid.dispose();

  return bbox;
}

const createModel = (modelUrl, anchorsUrl, classNamesUrl) => {
  const modelPromise = tf.loadLayersModel(modelUrl);
  const anchorsPromise = fetch(anchorsUrl).then((response) => response.json());
  const classNamesPromise = fetch(classNamesUrl).then((x) => x.text());

  const promise = Promise.all([
    modelPromise,
    anchorsPromise,
    classNamesPromise,
  ]);
  return promise;
};

// export { YoloV3 };
// const yolo = {
//   YoloPredictor: YoloPredictor,
// };

// module.exports = yolo;
const streamer = {YoloV3: YoloV3, createModel: createModel} 
module.exports = yolov3
